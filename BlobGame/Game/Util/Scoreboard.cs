using BlobGame.App;
using BlobGame.Game.GameModes;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace BlobGame.Game.Util;
/// <summary>
/// Class to keep track of the game's scores. Also handles saving and loading of the scores to the disk.
/// </summary>
public sealed class Scoreboard {
    private const string SCOREBOARD_API_ENDPOINT = "https://robotino.ch/toasted/api";
    private const string USER_AGENT = "Toasted! (https://github.com/gartoks/BlobGame)";
    /// <summary>
    /// The highest score ever achieved.
    /// </summary>
    private Dictionary<Guid, (int global, int[] daily)> Highscores { get; }

    /// <summary>
    /// Static constructor to initialize the score values.
    /// </summary>
    public Scoreboard() {
        Highscores = new Dictionary<Guid, (int global, int[] daily)>();
    }

    /// <summary>
    /// Loads and parses the scoreboard from the disk. If the file does not exist or is corrupted, the scoreboard is not loaded and scores are initialized to zero.
    /// </summary>
    internal void Load() {
        string file = Files.GetConfigFilePath("scores.sav");

        if (!File.Exists(file))
            return;

        Highscores.Clear();

        ScoreboardData[]? scoreDatas = JsonSerializer.Deserialize<ScoreboardData[]>(File.ReadAllText(file));

        if (scoreDatas == null)
            return;

        foreach (ScoreboardData scoreData in scoreDatas) {
            int globalHighscore = scoreData.GlobalHighscore;
            DateTime date = scoreData.Date;
            int[] dailyHighscores = scoreData.DailyHighscores;

            bool isDateValid = date.Date == DateTime.Today;
            for (int i = 0; i < dailyHighscores.Length; i++) {
                dailyHighscores[i] = isDateValid ? dailyHighscores[i] : 0;
            }

            Highscores.Add(scoreData.gameModeId, (globalHighscore, dailyHighscores));
        }
    }

    internal int GetGlobalHighscore(IGameMode gameMode) {
        if (Highscores.TryGetValue(gameMode.Id, out (int global, int[] daily) score))
            return score.global;
        return 0;
    }

    internal IReadOnlyList<int> GetDailyHighscores(IGameMode gameMode) {
        if (Highscores.TryGetValue(gameMode.Id, out (int global, int[] daily) score))
            return score.daily;
        return new int[3];
    }

    /// <summary>
    /// Adds a score to the score board. This method determines if the score is a new global highscore or a new daily highscore.
    /// Also saves the score board to the disk.
    /// </summary>
    /// <param name="score"></param>
    internal void AddScore(IGameMode gameMode, int score) {
        if (!Highscores.TryGetValue(gameMode.Id, out (int global, int[] daily) scores)) {
            scores = (0, new int[3]);
            Highscores.Add(gameMode.Id, scores);
        }

        if (score > scores.global)
            scores.global = score;

        for (int i = 0; i < scores.daily.Length; i++) {
            if (score > scores.daily[i]) {
                int lastScore = scores.daily[i];
                scores.daily[i] = score;

                score = lastScore;
            }
        }

        Highscores[gameMode.Id] = scores;

        Save();
    }

    /// <summary>
    /// Resets all scores and the highscore.
    /// </summary>
    public void Reset() {
        Highscores.Clear();
        Save();
    }

    /// <summary>
    /// Saves the scores to the disk.
    /// </summary>
    private void Save() {
        string file = Files.GetConfigFilePath("scores.sav");

        List<ScoreboardData> scoreboardDatas = new List<ScoreboardData>();
        foreach (KeyValuePair<Guid, (int global, int[] daily)> item in Highscores) {
            ScoreboardData scoreData = new ScoreboardData(item.Key, item.Value.global, DateTime.Today, item.Value.daily);
            scoreboardDatas.Add(scoreData);
        }

        File.WriteAllText(file, JsonSerializer.Serialize(scoreboardDatas.ToArray()));
    }

    internal int? SumbitScore(IGameMode gamemode, int score){
        if (DiscordAuth.IsSignedIn){
            var client = new HttpClient();
            var request = new HttpRequestMessage();
            request.RequestUri = new Uri(SCOREBOARD_API_ENDPOINT + "/scores");
            request.Method = HttpMethod.Post;

            request.Headers.Add("Accept", "*/*");
            request.Headers.Add("User-Agent", USER_AGENT);
            request.Headers.Add("Authorization", $"Bearer {DiscordAuth.GetTokens().AccessToken}");

            string gamemodeName = IGameMode.GameModeNames[gamemode.GetType()];

            var bodyString = $@"{{
                ""score"": {score},
                ""time"": ""{DateTime.Now.ToUniversalTime().ToString("o")}"",
                ""gamemode"": ""{gamemodeName.ToLower()}""
            }}";
            var content = new StringContent(bodyString, Encoding.UTF8, "application/json");
            request.Content = content;

            var response = client.SendAsync(request);
            response.Wait();
            var result = response.Result.Content.ReadFromJsonAsync<JsonNode>();
            result.Wait();
            
            Console.WriteLine(result.Result);
            return 0;
        }
        else{
            return null;
        }
    }

    private record ScoreboardData(Guid gameModeId, int GlobalHighscore, DateTime Date, int[] DailyHighscores);
}
