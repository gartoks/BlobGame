using BlobGame.Audio;
using BlobGame.Drawing;
using BlobGame.Game.Blobs;
using BlobGame.Game.GameControllers;
using BlobGame.Game.GameModes;
using BlobGame.Game.Gui;
using BlobGame.Game.Tutorial;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Scenes;
internal sealed class GameScene : Scene {
    private const float ARENA_OFFSET_X = Application.BASE_WIDTH * 0.5f;
    private const float ARENA_OFFSET_Y = 150;
    private const float DROP_INDICATOR_WIDTH = 10;

    internal IGameController Controller { get; }
    internal IGameMode Game { get; private set; }

    private TextureResource RankupArrowTexture { get; set; }
    private TextureResource ArenaBackgroundTexture { get; set; }
    private TextureResource ArenaBoxTexture { get; set; }
    private TextureResource MarkerTexture { get; set; }
    private TextureResource DropperTexture { get; set; }
    private NPatchTextureResource CurrentScoreTexture { get; set; }
    private TextureAtlasResource CurrentScoreBitmapFont { get; set; }
    private TextureResource ScoresBackgroundTexture { get; set; }
    private TextureResource CurrentBlobTexture { get; set; }
    private TextureResource NextBlobTexture { get; set; }
    private TextureResource HeldBlobTexture { get; set; }

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
            Application.BASE_WIDTH * 0.37f, Application.BASE_HEIGHT * 0.625f,
            Application.BASE_WIDTH * 0.2f, 100,
            "Retry",
            new Vector2(0.5f, 0.5f));
        ToMainMenuButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.62f, Application.BASE_HEIGHT * 0.625f,
            Application.BASE_WIDTH * 0.2f, 100,
            "To Menu",
            new Vector2(0.5f, 0.5f));
        GameOverPanel = new GuiPanel("0.5 0.5 1100px 500px", new Vector2(0.5f, 0.5f));
        GameOverLabel = new GuiLabel("0.5 0.35 1100px 120px", "Game over", new Vector2(0.5f, 0.5f));

        if (Application.Settings.IsTutorialEnabled)
            Tutorial = new TutorialDisplay();
    }

    /// <summary>
    /// Called when the scene is loaded. Override this method to provide custom scene initialization logic and to load resources.
    /// </summary>
    internal override void Load() {
        LoadAllGuiElements();

        Game.Load();
        Controller.Load();

        // Loads all the blob textures

        foreach (BlobData blobType in Game.Blobs.Values) {
            ResourceManager.TextureLoader.Load($"{blobType.Id}");
            ResourceManager.TextureLoader.Load($"{blobType.Id}_shadow");
        }

        RankupArrowTexture = ResourceManager.TextureLoader.Get("rankup_arrow");
        ArenaBackgroundTexture = ResourceManager.TextureLoader.Get("arena_bg");
        ArenaBoxTexture = ResourceManager.TextureLoader.Get("arena_box");
        MarkerTexture = ResourceManager.TextureLoader.Get("marker");
        DropperTexture = ResourceManager.TextureLoader.Get("dropper");
        CurrentScoreTexture = ResourceManager.NPatchTextureLoader.Get("currentScore_bg");
        CurrentScoreBitmapFont = ResourceManager.TextureAtlasLoader.Get("score_spritefont");
        ScoresBackgroundTexture = ResourceManager.TextureLoader.Get("scores_bg");
        CurrentBlobTexture = ResourceManager.TextureLoader.Get(Game.Blobs[Game.CurrentBlob].TextureKey);
        NextBlobTexture = ResourceManager.TextureLoader.Get(Game.Blobs[Game.NextBlob].TextureKey);
        HeldBlobTexture = ResourceManager.TextureLoader.Get(Game.Blobs[Game.NextBlob].TextureKey);

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
                Application.Settings.IsTutorialEnabled = false;
        } else {
            Game.Update(dT);
            Controller.Update(dT, Game);

            if (Game.CanSpawnBlob) {
                CurrentBlobTexture = ResourceManager.TextureLoader.Get(Game.Blobs[Game.CurrentBlob].TextureKey);
                NextBlobTexture = ResourceManager.TextureLoader.Get(Game.Blobs[Game.NextBlob].TextureKey);
            } else {
                CurrentBlobTexture = ResourceManager.TextureLoader.Fallback;
                NextBlobTexture = ResourceManager.TextureLoader.Get(Game.Blobs[Game.CurrentBlob].TextureKey);
            }

            if (Game.CanSpawnBlob && Controller.SpawnBlob(Game, out float t)) {
                t = Math.Clamp(t, 0, 1);
                Game.TrySpawnBlob(t);
            } else if (Controller.HoldBlob()) {
                Game.HoldBlob();
                CurrentBlobTexture = ResourceManager.TextureLoader.Get(Game.Blobs[Game.CurrentBlob].TextureKey);
                NextBlobTexture = ResourceManager.TextureLoader.Get(Game.Blobs[Game.NextBlob].TextureKey);
                HeldBlobTexture = ResourceManager.TextureLoader.Get(Game.Blobs[Game.HeldBlob].TextureKey);
            }
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

        DrawArenaBackground();

        Game.GameObjects.Enumerate(item => item.Draw());

        float t = Tutorial != null && !Tutorial.IsFinished ? 0.5f : Math.Clamp(Controller.GetCurrentT(), 0, 1);
        float indicatorOffset = DROP_INDICATOR_WIDTH / 2f + 1;
        float x = -IGameMode.ARENA_WIDTH / 2f + indicatorOffset + t * (IGameMode.ARENA_WIDTH - 2 * indicatorOffset);

        if (Game.IsGameOver)
            x = LastDropIndicatorX;
        else
            LastDropIndicatorX = x;

        DrawNextBlob();

        if (Game is not ClassicGameMode)
            DrawHeldBlob();

        if (Game.CanSpawnBlob) {
            DrawDropIndicator(x);
            DrawCurrentBlob(x);
        }
        DrawDropper(x);

        DrawArenaBox();

        RlGl.rlPopMatrix();

        Tutorial?.Draw();

        if (Game.IsGameOver)
            DrawGameOverScreen();
    }

    /// <summary>
    /// Called when the scene is about to be unloaded or replaced by another scene. Override this method to provide custom cleanup or deinitialization logic and to unload resources.
    /// </summary>
    internal override void Unload() {
        GameManager.Scoreboard.AddScore(Game, Game.Score);
        Controller.Close();
        // TODO unload NOT NEEDED resources
    }

    /// <summary>
    /// Called when two blobs combine.
    /// </summary>
    /// <param name="newType">The type of newly created blob.</param>
    private void Game_OnBlobsCombined(IGameMode sender, int newType) {
        AudioManager.PlaySound("piece_combination");
    }

    private void Game_OnGameOver(IGameMode sender) {
        if (sender.Score > GameManager.Scoreboard.GetGlobalHighscore(sender))
            AudioManager.PlaySound("new_highscore");
        else
            AudioManager.PlaySound("game_loss");
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
            new Rectangle(cX, cY, ruaW * 1.3f, ruaH * 1.3f),
            new Vector2(ruaW * 1.3f / 2, ruaH * 1.3f / 2),
            0,
            new Color(255, 255, 255, 255));
        //new Color(234, 89, 203, 255));

        int i = 0;
        foreach (BlobData blobType in Game.Blobs.Values) {
            float angle = (i + 1) / (float)(Game.Blobs.Count + 1) * MathF.Tau - MathF.PI / 2f;

            float x = cX + radius * MathF.Cos(angle);
            float y = cY + radius * MathF.Sin(angle);

            string texKey = Game.Blobs[blobType.Id].TextureKey;
            Texture tex = ResourceManager.TextureLoader.Get($"{texKey}_shadow").Resource;
            float w = tex.width;
            float h = tex.height;

            Raylib.DrawTexturePro(
                tex,
                new Rectangle(0, 0, w, h),
                new Rectangle(x, y, size, size),
                new Vector2(size / 2, size / 2), 0, Raylib.WHITE);
            i++;
        }
    }

    internal void DrawArenaBackground() {
        float w = ArenaBackgroundTexture.Resource.width;
        float h = ArenaBackgroundTexture.Resource.height;

        //ArenaTexture.Draw(new Rectangle(0, 0, 695, 858), new Vector2(w / 2f, 0));

        Raylib.DrawTexturePro(
            ArenaBackgroundTexture.Resource,
            new Rectangle(0, 0, w, h),
            new Rectangle(0, 0, w, h),
            new Vector2(w / 2, 0),
            0,
            Raylib.WHITE);
    }

    internal void DrawArenaBox() {
        float w = ArenaBoxTexture.Resource.width;
        float h = ArenaBoxTexture.Resource.height;

        //ArenaTexture.Draw(new Rectangle(0, 0, 695, 858), new Vector2(w / 2f, 0));

        Raylib.DrawTexturePro(
            ArenaBoxTexture.Resource,
            new Rectangle(0, 0, w, h),
            new Rectangle(0, 0, w, h),
            new Vector2(w / 2, 0),
            0,
            Raylib.WHITE);
    }

    internal void DrawNextBlob() {
        // Hightlight
        MarkerTexture.Draw(new Vector2(IGameMode.ARENA_WIDTH * 0.75f, 0), new Vector2(0.5f, 0.5f), null, 0, ResourceManager.ColorLoader.Get("light_accent").Resource);

        // Blob
        NextBlobTexture.Draw(
            new Vector2(IGameMode.ARENA_WIDTH * 0.75f, 0),
            new Vector2(0.5f, 0.5f),
            new Vector2(0.25f, 0.25f));

        Vector2 textPos = new Vector2(IGameMode.ARENA_WIDTH * 0.75f, -310);
        Raylib.DrawTextPro(
            Renderer.MainFont.Resource,
            "NEXT",
            textPos,
            textPos / 2f,
            -25.5f,
            80, 5, ResourceManager.ColorLoader.Get("font_dark").Resource);
    }

    internal void DrawHeldBlob() {
        // Hightlight
        MarkerTexture.Draw(new Vector2(IGameMode.ARENA_WIDTH * -0.75f, 0), new Vector2(0.5f, 0.5f), null, 0, ResourceManager.ColorLoader.Get("light_accent").Resource);

        if (Game.HeldBlob != -1) {
            // Blob
            HeldBlobTexture.Draw(
                new Vector2(IGameMode.ARENA_WIDTH * -0.75f, 0),
                new Vector2(0.5f, 0.5f),
                new Vector2(0.25f, 0.25f));
        }

        Vector2 textPos = new Vector2(IGameMode.ARENA_WIDTH * -1.85f, -280);
        Raylib.DrawTextPro(
            Renderer.MainFont.Resource,
            "HELD",
            textPos,
            textPos / 2f,
            0,
            80, 5, ResourceManager.ColorLoader.Get("font_dark").Resource);
    }

    internal void DrawDropper(float x) {
        DropperTexture.Draw(
            new Rectangle(x, 3f * IGameMode.ARENA_SPAWN_Y_OFFSET, 220, 150),
            new Vector2(0.5f, 0.5f));
    }

    internal void DrawDropIndicator(float x) {
        Raylib.DrawRectanglePro(
            new Rectangle(x, 0, DROP_INDICATOR_WIDTH, IGameMode.ARENA_HEIGHT),
            new Vector2(DROP_INDICATOR_WIDTH / 2f, 0),
            0,
            ResourceManager.ColorLoader.Get("background").Resource.ChangeAlpha(128));
    }

    internal void DrawCurrentBlob(float x) {
        CurrentBlobTexture.Draw(
            new Vector2(x, IGameMode.ARENA_SPAWN_Y_OFFSET),
            new Vector2(0.5f, 0.5f),
            new Vector2(0.25f, 0.25f),
            Game.SpawnRotation * RayMath.RAD2DEG);
    }

    private void DrawScoreboard() {
        const float x = 122.5f;
        const float y = 350;
        const float w = 400;

        DrawCurrentScore(x, y - 100, w);
        DrawHighscores(x, y + 120, w);
    }

    private void DrawCurrentScore(float x, float y, float w) {
        CurrentScoreBitmapFont.DrawAsBitmapFont(Game.Score.ToString(), 10, 120, new Vector2(x + w * 0.85f, y), new Vector2(1, 0));
    }

    private void DrawHighscores(float x, float y, float w) {
        ScoresBackgroundTexture.Draw(
            new Rectangle(x - w * 0.4f, y - 7, w * 1.6f, w * 1.65f));

        DrawScoreValue(x, y, w, GameManager.Scoreboard.GetGlobalHighscore(Game), "font_dark");

        for (int i = 0; i < GameManager.Scoreboard.GetDailyHighscores(Game).Count; i++) {
            DrawScoreValue(x, y + 160 + 130 * i, w, GameManager.Scoreboard.GetDailyHighscores(Game)[i]);
        }
    }

    private void DrawScoreValue(float x, float y, float w, int score, string colorKey = null, bool useMainFont = false, float fontSize = 90) {
        if (colorKey == null)
            colorKey = "font_light";

        FontResource font = useMainFont ? Renderer.MainFont : Renderer.GuiFont;

        string scoreStr = $"{score}";
        Vector2 scoreTextSize = Raylib.MeasureTextEx(font.Resource, scoreStr, 100, 10);
        Raylib.DrawTextPro(
                font.Resource,
                scoreStr,
                new Vector2(x + w - 50 - scoreTextSize.X, y + 5),
                new Vector2(scoreTextSize.Y / 2f, 0),
                0,
                fontSize,
                10,
                ResourceManager.ColorLoader.Get(colorKey).Resource);
    }

    private void DrawGameOverScreen() {
        GameOverPanel.Draw();
        GameOverLabel.Draw();

        RetryButton.Draw();
        ToMainMenuButton.Draw();

        bool isNewHighscore = Game.Score > GameManager.Scoreboard.GetGlobalHighscore(Game);
        GuiLabel ScoreLabel = new GuiLabel("0.5 0.45 1100px 90px",
            $"{(isNewHighscore ? "New Highscore!\n" : "")}Score: {Game.Score}",
            new Vector2(0.5f, 0.5f));
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
        float x = pos.X / Application.WorldToScreenMultiplierX - ARENA_OFFSET_X + IGameMode.ARENA_WIDTH / 2;
        float y = pos.Y / Application.WorldToScreenMultiplierY - ARENA_OFFSET_Y;
        return new Vector2(x, y);
    }
}
