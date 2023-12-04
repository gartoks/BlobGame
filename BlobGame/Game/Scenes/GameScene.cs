using BlobGame.App;
using BlobGame.Audio;
using BlobGame.Game.Blobs;
using BlobGame.Game.GameControllers;
using BlobGame.Game.GameModes;
using BlobGame.Game.Gui;
using BlobGame.Game.Tutorial;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;
using SimpleGL.Util.Math;
using System.Diagnostics;

namespace BlobGame.Game.Scenes;
internal sealed class GameScene : Scene {
    private const float ARENA_OFFSET_X = GameApplication.PROJECTION_WIDTH * 0.5f;
    private const float ARENA_OFFSET_Y = 150;
    private const float DROP_INDICATOR_WIDTH = 10;
    internal static Vector2 ARENA_OFFSET = new Vector2(ARENA_OFFSET_X, ARENA_OFFSET_Y);

    internal IGameController Controller { get; }
    internal IGameMode Game { get; private set; }

    private Texture TitleTexture { get; set; }
    private Texture RankupArrowTexture { get; set; }
    private Texture ArenaTexture { get; set; }
    private Texture MarkerTexture { get; set; }
    private Texture DropperTexture { get; set; }
    private Texture? CurrentBlobTexture { get; set; }
    private Texture NextBlobTexture { get; set; }

    private GuiPanel GameOverPanel { get; }
    private GuiLabel GameOverLabel { get; }
    private GuiTextButton RetryButton { get; }
    private GuiTextButton ToMainMenuButton { get; }
    private float LastDropIndicatorX { get; set; }

    private TutorialDisplay? Tutorial { get; }

    /// <summary>
    /// Creates a new game scene.
    /// </summary>
    public GameScene(IGameController controller, IGameMode gameMode) {
        Controller = controller;
        Game = gameMode;
        Game.OnBlobsCombined += Game_OnBlobsCombined;
        Game.OnGameOver += Game_OnGameOver;

        RetryButton = new GuiTextButton(
            GameApplication.PROJECTION_WIDTH * 0.37f, GameApplication.PROJECTION_HEIGHT * 0.625f,
            GameApplication.PROJECTION_WIDTH * 0.2f, 100,
            "Retry",
            10,
            new Vector2(0.5f, 0.5f));
        ToMainMenuButton = new GuiTextButton(
            GameApplication.PROJECTION_WIDTH * 0.62f, GameApplication.PROJECTION_HEIGHT * 0.625f,
            GameApplication.PROJECTION_WIDTH * 0.2f, 100,
            "To Menu",
            10,
            new Vector2(0.5f, 0.5f));
        GameOverPanel = new GuiPanel("0.5 0.5 1100px 500px", 9, new Vector2(0.5f, 0.5f));
        GameOverLabel = new GuiLabel("0.5 0.35 1100px 120px", "Game over", 10, new Vector2(0.5f, 0.5f));

        if (GameApplication.Settings.IsTutorialEnabled)
            Tutorial = new TutorialDisplay();
    }

    /// <summary>
    /// Called when the scene is loaded. Override this method to provide custom scene initialization logic and to load resources.
    /// </summary>
    internal override void Load() {
        // Loads all the blob textures
        for (int i = 0; i <= 10; i++) {
            ResourceManager.TextureLoader.GetResource($"{i}");
            ResourceManager.TextureLoader.GetResource($"{i}_shadow");
        }

        LoadAllGuiElements();

        Game.Load();
        Controller.Load();

        TitleTexture = ResourceManager.TextureLoader.GetResource("title_logo");
        RankupArrowTexture = ResourceManager.TextureLoader.GetResource("rankup_arrow");
        ArenaTexture = ResourceManager.TextureLoader.GetResource("arena_bg");
        MarkerTexture = ResourceManager.TextureLoader.GetResource("marker");
        DropperTexture = ResourceManager.TextureLoader.GetResource("dropper");
        CurrentBlobTexture = ResourceManager.TextureLoader.GetResource($"{(int)Game.CurrentBlob}");
        NextBlobTexture = ResourceManager.TextureLoader.GetResource($"{(int)Game.NextBlob}");

        Tutorial?.Load();
    }

    /// <summary>
    /// Called every frame to update the scene's state. 
    /// </summary>
    /// <param name="dT">The delta time since the last frame, typically used for frame-rate independent updates.</param>
    internal override void Update(float dT) {
        if (Tutorial != null && !Tutorial.IsFinished) {
            Tutorial.Update(dT);

            if (Tutorial.IsFinished)
                GameApplication.Settings.IsTutorialEnabled = false;
        } else {
            Game.Update(dT);
            Controller.Update(dT, Game);

            if (Game.CanSpawnBlob) {
                CurrentBlobTexture = ResourceManager.TextureLoader.GetResource($"{(int)Game.CurrentBlob}");
                NextBlobTexture = ResourceManager.TextureLoader.GetResource($"{(int)Game.NextBlob}");
            } else {
                CurrentBlobTexture = null;
                NextBlobTexture = ResourceManager.TextureLoader.GetResource($"{(int)Game.CurrentBlob}");
            }

            if (Game.CanSpawnBlob && Controller.SpawnBlob(Game, out float t)) {
                t = Math.Clamp(t, 0, 1);
                Game.TrySpawnBlob(t);
            }
        }
    }

    /// <summary>
    /// Called every frame to draw the scene. Override this method to provide custom scene rendering logic.
    /// </summary>
    internal override void Render() {
        DrawScoreboard();
        DrawRankupChart();

        DrawArena();
        DrawTitle();

        Game.GameObjects.Enumerate(item => item.Render(ARENA_OFFSET));

        float t = Tutorial != null && !Tutorial.IsFinished ? 0.5f : Math.Clamp(Controller.GetCurrentT(), 0, 1);
        float indicatorOffset = DROP_INDICATOR_WIDTH / 2f + 1;
        float x = -ClassicGameMode.ARENA_WIDTH / 2f + indicatorOffset + t * (ClassicGameMode.ARENA_WIDTH - 2 * indicatorOffset);

        if (Game.IsGameOver)
            x = LastDropIndicatorX;
        else
            LastDropIndicatorX = x;

        DrawNextBlob();

        if (Game.CanSpawnBlob) {
            DrawDropIndicator(x);
            DrawCurrentBlob(x);
        }
        DrawDropper(x);

        Tutorial?.Draw();

    }

    internal override void RenderGui() {
        base.RenderGui();


        if (Game.IsGameOver)
            DrawGameOverScreen();
    }

    /// <summary>
    /// Called when the scene is about to be unloaded or replaced by another scene. Override this method to provide custom cleanup or deinitialization logic and to unload resources.
    /// </summary>
    internal override void Unload() {
        GameManager.Scoreboard.AddScore(Game.Score);
        Controller.Close();
        // TODO unload NOT NEEDED resources
    }

    /// <summary>
    /// Called when two blobs combine.
    /// </summary>
    /// <param name="newType">The type of newly created blob.</param>
    private void Game_OnBlobsCombined(IGameMode sender, eBlobType newType) {
        AudioManager.PlaySound("piece_combination");
        Debug.WriteLine($"Blobs combined. New type: {newType}");
    }

    internal void DrawTitle() {
        Primitives.DrawSprite(new Vector2(-970, -35) + ARENA_OFFSET, new Vector2(400, 200), Vector2.Zero, -12.5f.ToRad(), 10, TitleTexture, Color4.White);
    }

    private void Game_OnGameOver(IGameMode sender) {
        if (sender.Score > GameManager.Scoreboard.GlobalHighscore)
            AudioManager.PlaySound("new_highscore");
        else
            AudioManager.PlaySound("game_loss");
    }

    internal void DrawRankupChart() {
        const float size = 75;
        const float radius = 200;

        const float cX = 1640;
        const float cY = 635f;

        Primitives.DrawSprite(new Vector2(cX, cY), new Vector2(), new Vector2(0.5f, 0.5f), 0, 10, RankupArrowTexture, Color4.White);

        int numBlobTypes = Enum.GetValues<eBlobType>().Length;
        for (int i = 0; i < numBlobTypes; i++) {
            float angle = (i + 1) / (float)(numBlobTypes + 1) * MathF.Tau - MathF.PI / 2f;

            float x = cX + radius * MathF.Cos(angle);
            float y = cY + radius * MathF.Sin(angle);

            Texture tex = ResourceManager.TextureLoader.GetResource($"{i}_shadow");

            Primitives.DrawSprite(new Vector2(x, y), new Vector2(size, size), new Vector2(0.5f, 0.5f), 0, 10, tex, Color4.White);
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
        Primitives.DrawSprite(new Vector2(ARENA_OFFSET_X, ARENA_OFFSET_Y), new Vector2(695, 858), new Vector2(0.5f, 0), 0, 3, ArenaTexture, Color4.White);
    }

    internal void DrawNextBlob() {
        float mW = MarkerTexture.Width;
        float mH = MarkerTexture.Height;

        // Hightlight
        Primitives.DrawSprite(new Vector2(ClassicGameMode.ARENA_WIDTH * 0.75f, 0) + ARENA_OFFSET, new Vector2(100, 100), new Vector2(0.5f, 0.5f), 0, 2, MarkerTexture, ResourceManager.ColorLoader.GetResource("light_accent"));

        // Blob
        (string name, eBlobType type, int score, float radius, float mass, string textureKey) nextBlob = BlobData.Data.Single(d => d.type == Game.NextBlob);
        Primitives.DrawSprite(new Vector2(ClassicGameMode.ARENA_WIDTH * 0.75f, 0) + ARENA_OFFSET, new Vector2(nextBlob.radius, nextBlob.radius), new Vector2(0.5f, 0.5f), 0, 3, NextBlobTexture, Color4.White);

        Vector2 textPos = new Vector2(ClassicGameMode.ARENA_WIDTH * 0.75f, -310);
        MeshFont font = Fonts.GetMainFont(80);
        Primitives.DrawText(font, "NEXT", ResourceManager.ColorLoader.GetResource("dark_accent"), textPos + ARENA_OFFSET, new Vector2(0.5f, 0.5f), -25.5f.ToRad(), 4);
    }

    internal void DrawDropper(float x) {
        Primitives.DrawSprite(new Vector2(x + 30, 3.5f * ClassicGameMode.ARENA_SPAWN_Y_OFFSET) + ARENA_OFFSET, new Vector2(192, 192), new Vector2(1 / 3f, 0.5f), 0, 6, DropperTexture, Color4.White);
    }

    internal void DrawDropIndicator(float x) {
        Primitives.DrawRectangle(new Vector2(x, 0) + ARENA_OFFSET, new Vector2(DROP_INDICATOR_WIDTH, ClassicGameMode.ARENA_HEIGHT), new Vector2(0.5f, 0), 0, 4, ResourceManager.ColorLoader.GetResource("background").ChangeAlpha(128));
    }

    internal void DrawCurrentBlob(float x) {
        if (CurrentBlobTexture == null)
            return;

        (string name, eBlobType type, int score, float radius, float mass, string textureKey) currentBlob = BlobData.Data.Single(d => d.type == Game.CurrentBlob);
        Primitives.DrawSprite(new Vector2(x, ClassicGameMode.ARENA_SPAWN_Y_OFFSET) + ARENA_OFFSET, new Vector2(currentBlob.radius, currentBlob.radius), new Vector2(0.5f, 0.5f), 0, 5, CurrentBlobTexture, Color4.White);
    }

    private void DrawCurrentScore(float x, float y, float w) {
        Primitives.DrawRectangleLines(new Vector2(x, y), new Vector2(w, 150), 8, new Vector2(0.5f, 0.5f), 0, 2, Color4.White);

        DrawScoreValue(x, y + 30, w, Game.Score);
    }

    private void DrawHighscores(float x, float y, float w) {
        Primitives.DrawRectangleLines(new Vector2(x, y), new Vector2(w, 550), 8, new Vector2(0.5f, 0.5f), 0, 2, Color4.White);

        Primitives.DrawLine(new Vector2(x, y + 125), new Vector2(x + w, y + 125), 8, 3, Color4.White);

        DrawScoreValue(x, y + 15, w, GameManager.Scoreboard.GlobalHighscore);
        for (int i = 0; i < GameManager.Scoreboard.DailyHighscores.Count; i++) {
            DrawScoreValue(x, y + 160 + 130 * i, w, GameManager.Scoreboard.DailyHighscores[i]);
        }
    }

    private void DrawScoreValue(float x, float y, float w, int score) {
        string scoreStr = $"{score}";

        MeshFont font = Fonts.GetMainFont(100);
        Vector2 scoreTextSize = font.MeasureText(scoreStr);
        Primitives.DrawText(font, scoreStr, ResourceManager.ColorLoader.GetResource("dark_accent"), new Vector2(x + w - 50 - scoreTextSize.X, y), new Vector2(0.5f, 0.5f), 0, 3);
    }

    private void DrawGameOverScreen() {
        GameOverPanel.Draw();
        GameOverLabel.Draw();

        RetryButton.Draw();
        ToMainMenuButton.Draw();

        bool isNewHighscore = Game.Score > GameManager.Scoreboard.GlobalHighscore;
        GuiLabel ScoreLabel = new GuiLabel("0.5 0.45 1100px 90px",
            $"{(isNewHighscore ? "New Highscore!\n" : "")}Score: {Game.Score}",
            8, new Vector2(0.5f, 0.5f));
        ScoreLabel.Draw();

        if (RetryButton.IsClicked)
            GameManager.SetScene(new GameScene(Controller, IGameMode.CreateGameMode(Game.GetType(), new Random().Next())));
        if (ToMainMenuButton.IsClicked)
            GameManager.SetScene(new MainMenuScene());
    }

    /// <summary>
    /// Converts a point in screen coordinates (such as the mouse position) to arena-localized coordinates with 0 at the arena's floor.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public static Vector2 ScreenToArenaPosition(Vector2 pos) {
        float x = pos.X/* / GameApplication.WorldToScreenMultiplierX*/ - ARENA_OFFSET_X + ClassicGameMode.ARENA_WIDTH / 2;
        float y = pos.Y/* / GameApplication.WorldToScreenMultiplierY*/ - ARENA_OFFSET_Y;
        return new Vector2(x, y);
    }
}
