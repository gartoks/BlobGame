using BlobGame.App;
using System.Text;

namespace BlobGame.Game;
internal static class Scoreboard {

    public static int GlobalHighscore { get; private set; }

    private static int[] _DailyHighscores { get; }
    public static IReadOnlyList<int> DailyHighscores => _DailyHighscores;

    static Scoreboard() {
        GlobalHighscore = 0;
        _DailyHighscores = new int[3];
    }

    internal static void Load() {
        string file = Files.GetSaveFilePath("highscores.txt");

        if (!File.Exists(file))
            return;

        string[] lines = File.ReadAllLines(file);
        if (lines.Length != 5)
            return;

        if (!int.TryParse(lines[0], out int globalHighscore))
            return;

        if (!DateTime.TryParse(lines[1], out DateTime date))
            return;

        if (!int.TryParse(lines[2], out int dailyScoreFirst))
            return;

        if (!int.TryParse(lines[3], out int dailyScoreSecond))
            return;

        if (!int.TryParse(lines[4], out int dailyScoreThird))
            return;

        GlobalHighscore = globalHighscore;

        if (date.Date != DateTime.Today) {
            _DailyHighscores[0] = 0;
            _DailyHighscores[1] = 0;
            _DailyHighscores[2] = 0;
        } else {
            _DailyHighscores[0] = dailyScoreFirst;
            _DailyHighscores[1] = dailyScoreSecond;
            _DailyHighscores[2] = dailyScoreThird;
        }
    }

    internal static void AddScore(int score) {
        if (score > GlobalHighscore)
            GlobalHighscore = score;

        for (int i = 0; i < _DailyHighscores.Length; i++) {
            if (score > _DailyHighscores[i]) {
                _DailyHighscores[i] = score;
                break;
            }
        }

        Save();
    }

    private static void Save() {
        string file = Files.GetSaveFilePath("highscores.txt");

        StringBuilder sb = new StringBuilder();
        sb.AppendLine(GlobalHighscore.ToString());
        sb.AppendLine(DateTime.Today.ToString());
        for (int i = 0; i < _DailyHighscores.Length; i++) {
            sb.AppendLine(_DailyHighscores[i].ToString());
        }

        File.WriteAllText(file, sb.ToString());
    }
}
