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

        ExpandoObject? tmp = JsonSerializer.Deserialize<ExpandoObject>(File.ReadAllText(file));

        if (tmp == null)
            return;

        dynamic scoreData = tmp;

        int globalHighscore = scoreData.GlobalHighscore;
        DateTime date = scoreData.Date;
        int[] dailyHighscores = scoreData.DailyHighscores;

        GlobalHighscore = globalHighscore;

        bool isDateValid = date.Date == DateTime.Today;
        for (int i = 0; i < _DailyHighscores.Length; i++) {
            _DailyHighscores[i] = isDateValid ? dailyHighscores[i] : 0;
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
                int lastScore = _DailyHighscores[i];
                _DailyHighscores[i] = score;

                score = lastScore;
            }
        }

        Save();
    }

    /// <summary>
    /// Resets all scores and the highscore.
    /// </summary>
    public void Reset() {
        GlobalHighscore = 0;
        for (int i = 0; i < _DailyHighscores.Length; i++) {
            _DailyHighscores[i] = 0;
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
