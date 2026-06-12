use std::fmt;

use tracing::{Event, Subscriber};
use tracing_subscriber::{
    fmt::{
        format::{self, FormatEvent, FormatFields},
        FmtContext,
    },
    registry::LookupSpan,
};

pub struct LogFormatter;

impl<S, N> FormatEvent<S, N> for LogFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'writer> FormatFields<'writer> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let meta = event.metadata();

        // Exact Mangal TUI Colors (ansi 256):
        // 62 = Indigo/Purple (Mangal primary Title background)
        // 230 = Cream/White (Mangal fg text for titles)
        // 1 = Red, 2 = Green, 3 = Yellow, 4 = Blue, 5 = Purple, 6 = Cyan

        // Using Mangal 'plain' icons since Nerd fonts are not rendering
        // Success: "✓" (Green)
        // Fail: "X" (Red)
        // Question: "?" (Yellow)
        // Progress: "@" (Blue)
        // Search: "S" (Cyan)
        let (bg, fg, label, icon, icon_col) = match *meta.level() {
            tracing::Level::TRACE => ("6", "230", " TRAC ", "S", "6"),
            tracing::Level::DEBUG => ("4", "230", " DBUG ", "@ ", "4"),
            tracing::Level::INFO => ("62", "230", " INFO ", "✓ ", "2"),
            tracing::Level::WARN => ("3", "230", " WARN ", "? ", "3"),
            tracing::Level::ERROR => ("1", "230", " FAIL ", "X ", "1"),
        };

        let rst = "\x1b[0m";

        // 1. Draw the log level as a Mangal TUI Title (bg/fg padded tag)
        write!(writer, "\x1b[38;5;{fg};48;5;{bg}m{label}{rst} ")?;

        // 2. Draw the corresponding Mangal UI icon with its specific color
        write!(writer, "\x1b[38;5;{icon_col}m{icon}{rst} ")?;

        // 3. Write fields (the log message)
        write!(writer, "\x1b[38;5;252m")?; // light grey text for body
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        write!(writer, "{rst}")?;

        writeln!(writer)
    }
}
