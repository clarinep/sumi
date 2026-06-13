use std::fmt;

use tracing::{Event, Level, Subscriber};
use tracing_subscriber::{
    fmt::{
        FmtContext,
        format::{self, FormatEvent, FormatFields},
    },
    registry::LookupSpan,
};

#[derive(Debug, Clone, Copy)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl Color {
    const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1b[38;2;{};{};{}m", self.r, self.g, self.b)
    }
}

const COLOR_TRACE: Color = Color::new(205, 180, 219);
const COLOR_DEBUG: Color = Color::new(189, 224, 254);
const COLOR_INFO: Color = Color::new(241, 138, 131);
const COLOR_WARN: Color = Color::new(255, 180, 162);
const COLOR_FAIL: Color = Color::new(239, 35, 60);
const COLOR_TEXT: Color = Color::new(226, 226, 226);

pub struct LogFormatter;

impl<S, N> FormatEvent<S, N> for LogFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'writer> FormatFields<'writer> + 'static,
{
    fn format_event(
        &self,
        context: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let metadata = event.metadata();
        let level = metadata.level();

        let (color, label) = match *level {
            Level::TRACE => (COLOR_TRACE, "trace"),
            Level::DEBUG => (COLOR_DEBUG, "debug"),
            Level::INFO => (COLOR_INFO, "info "),
            Level::WARN => (COLOR_WARN, "warn "),
            Level::ERROR => (COLOR_FAIL, "fail "),
        };

        let reset = "\x1b[0m";

        write!(writer, "{color}{label}{reset} ")?;
        write!(writer, "{COLOR_TEXT}")?;
        context.field_format().format_fields(writer.by_ref(), event)?;
        write!(writer, "{reset}")?;

        writeln!(writer)
    }
}
