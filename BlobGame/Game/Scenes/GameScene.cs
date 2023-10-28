using BlobGame.Drawing;
using BlobGame.Game.Blobs;
using BlobGame.Game.GameControllers;
using BlobGame.Game.GameObjects;
using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Scenes;
internal sealed class GameScene : Scene {
    private const float ARENA_OFFSET_X = Application.BASE_WIDTH * 0.5f;
    private const float ARENA_OFFSET_Y = 150;
    private Color DROP_INDICATOR_COLOR { get; } = new Color(255, 255, 255, 128);
    private const float DROP_INDICATOR_WIDTH = 10;

    internal IGameController Controller { get; }
    internal Simulation GameSim { get; private set; }

    private TextureResource TitleTexture { get; set; }
    private TextureResource RankupArrowTexture { get; set; }
    private TextureResource ArenaTexture { get; set; }
    private TextureResource MarkerTexture { get; set; }
    private TextureResource PlacerTexture { get; set; }
    private TextureResource CurrentBlobTexture { get; set; }
    private TextureResource NextBlobTexture { get; set; }

    private GUIPanel GameOverPanel { get; }
    private GUILabel GameOverLabel { get; }
    private GUITextButton RetryButton { get; }
    private GUITextButton ToMainMenuButton { get; }
    private float LastDropIndicatorX { get; set; }

    /// <summary>
    /// Creates a new game scene.
    /// </summary>
    public GameScene() {
        Controller = new MouseController(this);
        GameSim = new Simulation(new Random().Next());

        RetryButton = new GUITextButton(
            Application.BASE_WIDTH * 0.37f, Application.BASE_HEIGHT * 0.6f,
            Application.BASE_WIDTH * 0.2f, 100,
            "Retry",
            new Vector2(0.5f, 0.5f));
        ToMainMenuButton = new GUITextButton(
            Application.BASE_WIDTH * 0.62f, Application.BASE_HEIGHT * 0.6f,
            Application.BASE_WIDTH * 0.2f, 100,
            "To Menu",
            new Vector2(0.5f, 0.5f));
        GameOverPanel = new GUIPanel(
            new Vector2(Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT / 2f),
            new Vector2(1100, 500),
            Renderer.MELBA_LIGHT_PINK,
            new Vector2(0.5f, 0.5f));
        GameOverLabel = new GUILabel(
            new Vector2(Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.35f),
            new Vector2(1100, 120),
            "Game over",
            new Vector2(0.5f, 0.5f));
    }


    /// <summary>
    /// Called when the scene is loaded. Override this method to provide custom scene initialization logic and to load resources.
    /// </summary>
    internal override void Load() {
        Scoreboard.Load();

        // Loads all the blob textures
        for (int i = 0; i < 1; i++)
            ResourceManager.LoadTexture($"{i}");

        GameSim.Load();

        TitleTexture = ResourceManager.GetTexture("title_logo");
        RankupArrowTexture = ResourceManager.GetTexture("rankup_arrow");
        ArenaTexture = ResourceManager.GetTexture("arena_bg");
        MarkerTexture = ResourceManager.GetTexture("marker");
        PlacerTexture = ResourceManager.GetTexture("tutel");
        CurrentBlobTexture = ResourceManager.GetTexture($"{(int)GameSim.CurrentBlob}");
        NextBlobTexture = ResourceManager.GetTexture($"{(int)GameSim.NextBlob}");
    }

    /// <summary>
    /// Called every frame to update the scene's state. 
    /// </summary>
    /// <param name="dT">The delta time since the last frame, typically used for frame-rate independent updates.</param>
    internal override void Update(float dT) {
        GameSim.Update(dT);

        if (GameSim.CanSpawnBlob) {
            CurrentBlobTexture = ResourceManager.GetTexture($"{(int)GameSim.CurrentBlob}");
            NextBlobTexture = ResourceManager.GetTexture($"{(int)GameSim.NextBlob}");
        } else {
            CurrentBlobTexture = ResourceManager.DefaultTexture;
            NextBlobTexture = ResourceManager.GetTexture($"{(int)GameSim.CurrentBlob}");
        }

        if (Controller.SpawnBlob(GameSim, out float t) && GameSim.CanSpawnBlob) {
            t = Math.Clamp(t, 0, 1);
            GameSim.TrySpawnBlob(t, out Blob? blob);
        }
    }

    /// <summary>
    /// Called every frame to draw the scene. Override this method to provide custom scene rendering logic.
    /// </summary>
    internal override void Draw() {
        RlGl.rlPushMatrix();
        DrawScoreboard();
        DrawRankupChart();

        RlGl.rlTranslatef(ARENA_OFFSET_X, ARENA_OFFSET_Y, 0);

        DrawArena();
        DrawTitle();

        GameSim.GameObjects.Enumerate(item => item.Draw());

        float t = Math.Clamp(Controller.GetCurrentT(), 0, 1);
        float indicatorOffset = DROP_INDICATOR_WIDTH / 2f + 1;
        float x = -Simulation.ARENA_WIDTH / 2f + indicatorOffset + t * (Simulation.ARENA_WIDTH - 2 * indicatorOffset);

        if (GameSim.IsGameOver)
            x = LastDropIndicatorX;
        else
            LastDropIndicatorX = x;

        DrawNextBlob();

        if (GameSim.CanSpawnBlob) {
            DrawDropIndicator(x);
            DrawCurrentBlob(x);
        }
        DrawDropper(x);

        RlGl.rlPopMatrix();

        if (GameSim.IsGameOver)
            DrawGameOverScreen();
    }

    /// <summary>
    /// Called when the scene is about to be unloaded or replaced by another scene. Override this method to provide custom cleanup or deinitialization logic and to unload resources.
    /// </summary>
    internal override void Unload() {
        Scoreboard.AddScore(GameSim.Score);
        // TODO unload NOT NEEDED resources
    }

    /// <summary>
    /// Converts a point in screen coordinates (such as the mouse position) to arena-localized coordinates with 0 at the arena's floor.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public Vector2 ScreenToArenaPosition(Vector2 pos) {
        float x = pos.X / Application.WorldToScreenMultiplierX - ARENA_OFFSET_X + Simulation.ARENA_WIDTH / 2;
        float y = pos.Y / Application.WorldToScreenMultiplierY - ARENA_OFFSET_Y;
        return new Vector2(x, y);
    }

    internal void DrawTitle() {
        float w = TitleTexture.Resource.width;
        float h = TitleTexture.Resource.height;

        Raylib.DrawTexturePro(
            TitleTexture.Resource,
            new Rectangle(0, 0, w, h),
            new Rectangle(-970, -35, w * 0.7f, h * 0.7f),
            new Vector2(0, 0),
            -12.5f,
            Raylib.WHITE);
    }

    internal void DrawRankupChart() {
        const float size = 75;
        const float radius = 200;

        float ruaW = RankupArrowTexture.Resource.width;
        float ruaH = RankupArrowTexture.Resource.height;

        float cX = 1640;
        float cY = 635f;

        Raylib.DrawTexturePro(
            RankupArrowTexture.Resource,
            new Rectangle(0, 0, ruaW, ruaH),
            new Rectangle(cX, cY, ruaW, ruaH),
            new Vector2(ruaW / 2, ruaH / 2),
            0,
            new Color(255, 255, 255, 255));
        //new Color(234, 89, 203, 255));

        int numBlobTypes = Enum.GetValues<eBlobType>().Length;
        for (int i = 0; i < numBlobTypes; i++) {
            float angle = (i + 1) / (float)(numBlobTypes + 1) * MathF.Tau - MathF.PI / 2f;

            float x = cX + radius * MathF.Cos(angle);
            float y = cY + radius * MathF.Sin(angle);

            Texture tex = ResourceManager.GetTexture($"{i}").Resource;
            float w = tex.width;
            float h = tex.height;

            Raylib.DrawTexturePro(
                ResourceManager.GetTexture($"{i}").Resource,
                new Rectangle(0, 0, w, h),
                new Rectangle(x, y, size, size),
                new Vector2(size / 2, size / 2), 0, Raylib.WHITE);
        }
    }

    internal void DrawScoreboard() {
        const float x = 122.5f;
        const float y = 250;
        const float w = 400;

        DrawCurrentScore(x, y, w);
        DrawHighscores(x, y + 200, w);
    }

    internal void DrawArena() {
        float w = ArenaTexture.Resource.width;
        float h = ArenaTexture.Resource.height;

        Raylib.DrawTexturePro(
            ArenaTexture.Resource,
            new Rectangle(0, 0, w, h),
            new Rectangle(0, 0, w, h),
            new Vector2(w / 2, 0),
            0,
            Raylib.WHITE);
    }

    internal void DrawNextBlob() {
        float w = NextBlobTexture.Resource.width;
        float h = NextBlobTexture.Resource.height;

        float mW = MarkerTexture.Resource.width;
        float mH = MarkerTexture.Resource.height;

        // Hightlight
        Raylib.DrawTexturePro(
            MarkerTexture.Resource,
            new Rectangle(0, 0, mW, mH),
            new Rectangle(Simulation.ARENA_WIDTH * 0.75f, 0, mW, mH),
            new Vector2(mW / 2, mH / 2),
            0,
            Raylib.WHITE
            );

        // Blob
        Raylib.DrawTexturePro(
            NextBlobTexture.Resource,
            new Rectangle(0, 0, w, h),
            new Rectangle(Simulation.ARENA_WIDTH * 0.75f, 0, w, h),
            new Vector2(w / 2, h / 2),
            0,
            Raylib.WHITE);
    }

    internal void DrawDropper(float x) {
        float w = PlacerTexture.Resource.width;
        float h = PlacerTexture.Resource.height;

        Raylib.DrawTexturePro(
            PlacerTexture.Resource,
            new Rectangle(0, 0, w, h),
            new Rectangle(x, Simulation.ARENA_SPAWN_Y_OFFSET - 0.75f * h, 1.5f * w, 1.5f * h),
            new Vector2(w * 0.33f, h / 2),
            0,
            Raylib.WHITE);
    }

    internal void DrawDropIndicator(float x) {
        Raylib.DrawRectanglePro(
            new Rectangle(x, 0, DROP_INDICATOR_WIDTH, Simulation.ARENA_HEIGHT),
            new Vector2(DROP_INDICATOR_WIDTH / 2f, 0),
            0,
            DROP_INDICATOR_COLOR);
    }

    internal void DrawCurrentBlob(float x) {
        float w = CurrentBlobTexture.Resource.width;
        float h = CurrentBlobTexture.Resource.height;

        Raylib.DrawTexturePro(
            CurrentBlobTexture.Resource,
            new Rectangle(0, 0, w, h),
            new Rectangle(x, Simulation.ARENA_SPAWN_Y_OFFSET, w, h),
            new Vector2(w / 2, h / 2),
            0,
            Raylib.WHITE);
    }

    private void DrawCurrentScore(float x, float y, float w) {
        Raylib.DrawRectangleRoundedLines(
            new Rectangle(x, y, w, 150),
            0.3f, 10, 8, Raylib.WHITE);

        DrawScoreValue(x, y + 30, w, GameSim.Score);
    }

    private void DrawHighscores(float x, float y, float w) {
        Raylib.DrawRectangleRoundedLines(
            new Rectangle(x, y, w, 550),
            0.15f, 10, 8, Raylib.WHITE);

        Raylib.DrawLineEx(new Vector2(x, y + 125), new Vector2(x + w, y + 125), 8, Raylib.WHITE);

        DrawScoreValue(x, y + 15, w, Scoreboard.GlobalHighscore);
        for (int i = 0; i < Scoreboard.DailyHighscores.Count; i++) {
            DrawScoreValue(x, y + 160 + 130 * i, w, Scoreboard.DailyHighscores[i]);
        }
    }

    private void DrawScoreValue(float x, float y, float w, int score) {
        string scoreStr = $"{score}";
        Vector2 scoreTextSize = Raylib.MeasureTextEx(Renderer.Font.Resource, scoreStr, 100, 10);
        Raylib.DrawTextEx(
            Renderer.Font.Resource,
            scoreStr,
            new Vector2(x + w - 50 - scoreTextSize.X, y),
            100,
            10,
            Raylib.WHITE);
    }

    private void DrawGameOverScreen(){
        GameOverPanel.Draw();
        GameOverLabel.Draw();

        var ScoreLabel = new GUILabel(
            new Vector2(Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.45f),
            new Vector2(1100, 90),
            $"Score: {GameSim.Score}",
            new Vector2(0.5f, 0.5f));
        ScoreLabel.Draw();

        if (RetryButton.Draw())
            GameManager.SetScene(new GameScene());
        if (ToMainMenuButton.Draw())
            GameManager.SetScene(new MainMenuScene());
    }
}
