using BlobGame.App;
using System.Dynamic;
using System.Text.Json;

namespace BlobGame.Game.Util;
/// <summary>
/// Class to keep track of the game's scores. Also handles saving and loading of the scores to the disk.
/// </summary>
public sealed class Scoreboard {
    /// <summary>
    /// The highest score ever achieved.
    /// </summary>
    public int GlobalHighscore { get; private set; }

    /// <summary>
    /// The highest scores achieved today. The first element is the highest score, the second element is the second highest score and so on.
    /// Currently lists the top 3 scores.
    /// </summary>
    private int[] _DailyHighscores { get; }
    /// <summary>
    /// A read-only version of the daily highscores. Used for access outside of this class.
    /// </summary>
    public IReadOnlyList<int> DailyHighscores => _DailyHighscores;

    /// <summary>
    /// Static constructor to initialize the score values.
    /// </summary>
    public Scoreboard() {
        GlobalHighscore = 0;
        _DailyHighscores = new int[3];
    }

    /// <summary>
    /// Loads and parses the scoreboard from the disk. If the file does not exist or is corrupted, the scoreboard is not loaded and scores are initialized to zero.
    /// </summary>
    internal void Load() {
        string file = Files.GetConfigFilePath("scores.sav");

        if (!File.Exists(file))
            return;

        dynamic? scoreData = JsonSerializer.Deserialize<dynamic>(file);

        if (scoreData == null)
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
    internal void AddScore(int score) {
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

    /// <summary>
    /// Saves the scores to the disk.
    /// </summary>
    private void Save() {
        string file = Files.GetConfigFilePath("scores.sav");

        dynamic scoreData = new ExpandoObject();
        scoreData.GlobalHighscore = GlobalHighscore;
        scoreData.Date = DateTime.Today;
        scoreData.DailyHighscores = _DailyHighscores;

        File.WriteAllText(file, JsonSerializer.Serialize(scoreData));
    }
}
