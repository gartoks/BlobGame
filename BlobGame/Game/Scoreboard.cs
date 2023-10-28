using BlobGame.App;
using System.Text;

namespace BlobGame.Game;
/// <summary>
/// Static class to keep track of the game's scores. Also handles saving and loading of the scores to the disk.
/// </summary>
internal static class Scoreboard {
    /// <summary>
    /// The highest score ever achieved.
    /// </summary>
    public static int GlobalHighscore { get; private set; }

    /// <summary>
    /// The highest scores achieved today. The first element is the highest score, the second element is the second highest score and so on.
    /// Currently lists the top 3 scores.
    /// </summary>
    private static int[] _DailyHighscores { get; }
    /// <summary>
    /// A read-only version of the daily highscores. Used for access outside of this class.
    /// </summary>
    public static IReadOnlyList<int> DailyHighscores => _DailyHighscores;

    /// <summary>
    /// Static constructor to initialize the score values.
    /// </summary>
    static Scoreboard() {
        GlobalHighscore = 0;
        _DailyHighscores = new int[3];
    }

    /// <summary>
    /// Loads and parses the scoreboard from the disk. If the file does not exist or is corrupted, the scoreboard is not loaded and scores are initialized to zero.
    /// </summary>
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

    /// <summary>
    /// Adds a score to the score board. This method determines if the score is a new global highscore or a new daily highscore.
    /// Also saves the score board to the disk.
    /// </summary>
    /// <param name="score"></param>
    internal static void AddScore(int score) {
        if (score > GlobalHighscore)
            GlobalHighscore = score;

        for (int i = 0; i < _DailyHighscores.Length; i++) {
            if (score > _DailyHighscores[i]) {
                int lastScore = _DailyHighscores[i];
                _DailyHighscores[i] = score;

                score = lastScore;
            }
        }

        Save();
    }

    /// <summary>
    /// Saves the scores to the disk.
    /// </summary>
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
