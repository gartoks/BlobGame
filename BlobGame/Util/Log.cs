namespace BlobGame.Util;
public enum eLogType { Message, Warning, Error }

public delegate void LogEventHandler(string text, eLogType logType);

internal class Log {
    public static event LogEventHandler? OnLog;

    public static void WriteLine(string text, eLogType messageType = eLogType.Message) {
        text = $"{TimeTag()}: {text}";
        OnLog?.Invoke(text, messageType);
    }

    public static void WriteLineIf(bool condition, string text, eLogType messageType = eLogType.Message) {
        if (condition)
            WriteLine(text, messageType);
    }

    private static string TimeTag() {
        DateTime timestamp = DateTime.Now;
        return $"[{timestamp.Hour.ToString().PadLeft(2, '0')}:{timestamp.Minute.ToString().PadLeft(2, '0')}:{timestamp.Second.ToString().PadLeft(2, '0')}]";
    }
}