use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use moka::future::Cache;

use crate::renderer::{canvas::RawCardImage, error::RenderError};

// limit of images, moka will auto kick older images
// moka uses LFU algorithm so if by rng a card smh keeps getting dropped then it will protect
// and kick out other less "popular" card that was NATURALLY unlucky to not be chosen by blair.
// -- Buat lebih konteks per kartu sekitar 200kb, kalau RAM kena cap kita turunin aja
// -- Tapi kita sini pake raw rgba yg belum dikompres, sekitar 3 juta byte
const MAX_CACHE_SIZE_KB: u64 = 1_000_000; // -- 1 GB limit di kilobyte

#[derive(Default, Debug)]
#[repr(align(128))]
pub struct CachePadded<T> {
    pub value: T,
}

#[derive(Default, Debug)]
pub struct CacheStats {
    pub hits: CachePadded<std::sync::atomic::AtomicU64>,
    pub misses: CachePadded<std::sync::atomic::AtomicU64>,
}

/// hold cached image and list of file here
pub struct CardCache {
    memory: Cache<Arc<str>, Arc<RawCardImage>>,
    file_index: HashMap<Arc<str>, PathBuf>,
    pub stats: CacheStats,
}

impl std::fmt::Debug for CardCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CardCache")
            .field("memory", &self.memory)
            .field("file_index_len", &self.file_index.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl CardCache {
    /// sets up the cache and finds all images
    pub fn new(cards_directory: impl AsRef<std::path::Path>) -> Result<Self, &'static str> {
        let cache = Cache::builder()
            .max_capacity(MAX_CACHE_SIZE_KB)
            .weigher(|_key, value: &Arc<RawCardImage>| -> u32 {
                // -- Cache di kilobytes biar bisa gampang ganti ke value gede
                (value.pixels.len() / 1024) as u32
            })
            .build();

        let file_index = Self::build_card_list(cards_directory.as_ref());

        if file_index.is_empty() {
            return Err("found literally zero cards in the folder.. sumi is refusing to wake up");
        }

        log::info!("found {} card images on disk", file_index.len());

        Ok(Self { memory: cache, file_index, stats: CacheStats::default() })
    }

    /// makes a list of all image files in the folder
    /// we check this list first so we dont waste time looking for missing files.
    fn build_card_list(cards_dir: &Path) -> HashMap<Arc<str>, PathBuf> {
        fn find_card(base_dir: &Path, dir: &Path, index: &mut HashMap<Arc<str>, PathBuf>) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        find_card(base_dir, &path, index);
                    } else if path.extension().is_some_and(|e| e == "webp") {
                        if let Ok(rel_path) = path.strip_prefix(base_dir) {
                            let key_path = rel_path.with_extension("");
                            let name_str = key_path.to_string_lossy().replace('\\', "/");
                            index.insert(name_str.into(), path);
                        }
                    }
                }
            }
        }
        let mut index = HashMap::new();
        find_card(cards_dir, cards_dir, &mut index);
        index
    }

    /// starts a background task to lazily prewarm cards into memory
    pub fn start_prewarm(self: &Arc<Self>) {
        if self.file_index.is_empty() {
            return;
        }

        let file_index_clone = self.file_index.clone();
        let memory = self.memory.clone();

        // spawn background task to slowly warm the cache without freezing server
        tokio::spawn(async move {
            log::info!("baking moka cache..");
            let warmed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let warmed_kb = Arc::new(std::sync::atomic::AtomicU64::new(0));

            // limit cpu decodes to avoid blocking other async stuff and memory
            let semaphore = Arc::new(tokio::sync::Semaphore::new(8));

            for (name, path) in file_index_clone {
                // check if we reached ~90% of capacity
                let current_kb = warmed_kb.load(std::sync::atomic::Ordering::Relaxed);
                if current_kb > (MAX_CACHE_SIZE_KB * 9 / 10) {
                    log::info!("cache prewarm reached capped limit, stopping!");
                    break;
                }

                // fetch files on this single loop task to account for our shit HDD
                let Ok(file_bytes) = tokio::fs::read(&path).await else {
                    continue;
                };

                let memory = memory.clone();
                let warmed = warmed.clone();
                let warmed_kb = warmed_kb.clone();

                // again acquire permit AFTER reading the file, to limit cpu decoding work and memory
                let permit = semaphore.clone().acquire_owned().await.unwrap();

                tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        if let Ok((pixels, width, height)) = webpx::decode_rgba(&file_bytes) {
                            Some(Arc::new(crate::renderer::canvas::RawCardImage {
                                width,
                                height,
                                pixels,
                            }))
                        } else {
                            None
                        }
                    })
                    .await
                    .unwrap_or(None);

                    if let Some(arc_img) = result {
                        let size_kb = (arc_img.pixels.len() / 1024) as u64;
                        memory.insert(name, arc_img).await;
                        warmed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        warmed_kb.fetch_add(size_kb, std::sync::atomic::Ordering::Relaxed);
                    }

                    // manually run pending tasks so moka processes evictions accurately
                    memory.run_pending_tasks().await;

                    drop(permit);
                });
            }

            // wait for every damn remaining tasks to complete by acquiring all available permits
            let _ = semaphore.acquire_many(8).await.unwrap();

            // slightly innacurate size as it counts the lexend deca .ttf file
            log::info!(
                "finished baking {} cards - {} mb",
                warmed.load(std::sync::atomic::Ordering::Relaxed),
                warmed_kb.load(std::sync::atomic::Ordering::Relaxed) / 1024
            );
        });
    }
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, f64) {
        let hits = self.stats.hits.value.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.stats.misses.value.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 { 0.0 } else { (hits as f64 / total as f64) * 100.0 };
        (hits, misses, hit_rate)
    }

    /// gets an image from cache.
    /// -- Sekarang klo gak ada kartu di situ, kita load dari disk. akan lebih lemot.
    pub async fn get_card(&self, name: &str) -> Result<Arc<RawCardImage>, RenderError> {
        if let Some(img) = self.memory.get(name).await {
            self.stats.hits.value.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(img);
        }

        self.stats.misses.value.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // a check on whether file exists before trying to read it
        let path = self
            .file_index
            .get(name)
            .cloned()
            .ok_or_else(|| RenderError::CardNotFound(name.to_string()))?;

        let name_arc: Arc<str> = name.into();

        self.memory
            .try_get_with(name_arc, async move {
                tokio::task::spawn_blocking(move || {
                    // open the file and decode the image directly using webpx
                    let file_bytes = std::fs::read(&path).map_err(|e| {
                        RenderError::Internal(format!(
                            "failed to open file '{}': {e}",
                            path.display()
                        ))
                    })?;

                    let (pixels, width, height) = webpx::decode_rgba(&file_bytes).map_err(|e| {
                        RenderError::Internal(format!(
                            "failed to decode webp for '{}': {e:?}",
                            path.display()
                        ))
                    })?;

                    let image = RawCardImage { width, height, pixels };
                    Ok(Arc::new(image))
                })
                .await
                .unwrap_or_else(|e| Err(RenderError::Internal(format!("task panicked: {e}"))))
            })
            .await
            .map_err(|e| e.as_ref().clone())
    }
}
