using BlobGame.App;
using BlobGame.Audio;
using BlobGame.Game.Blobs;
using BlobGame.Game.GameControllers;
using BlobGame.Game.GameModes;
using BlobGame.Game.Gui;
using BlobGame.Game.Tutorial;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Scenes;
internal sealed partial class GameScene : Scene {
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
    private TextureResource CurrentScoreBackgroundTexture { get; set; }
    private TextureAtlasResource CurrentScoreBitmapFont { get; set; }
    private TextureResource ScoresBackgroundTexture { get; set; }
    private TextureResource GameOverTexture { get; set; }

    private Dictionary<Guid, (TextureResource tex, float alpha, float rotation, Vector2 position)> Splats { get; }

    private GuiNPatchPanel GameOverPanel { get; }
    private GuiNPatchPanel MenuPanel { get; }
    private GuiTextButton RetryButton { get; }
    private GuiLabel MenuLabel { get; }
    private GuiTextButton ContinueButton { get; }
    private GuiTextButton ToMainMenuButton { get; }
    private float LastDropIndicatorX { get; set; }

    private GuiTextButton MenuButton { get; }

    private TutorialDisplay? Tutorial { get; set; }

    private Random Random { get; }

    private bool IsTutorialEnabled => Tutorial != null && !Tutorial.IsFinished;
    private bool IsMenuOpen { get; set; }
    private bool IsPaused => IsMenuOpen || IsTutorialEnabled || Game.IsGameOver;

    /// <summary>
    /// Creates a new game scene.
    /// </summary>
    public GameScene(IGameController controller, IGameMode gameMode) {
        Controller = controller;
        Game = gameMode;
        Game.OnBlobsCombined += Game_OnBlobsCombined;
        Game.OnBlobDestroyed += Game_OnBlobDestroyed;
        Game.OnGameOver += Game_OnGameOver;

        RetryButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.37f, Application.BASE_HEIGHT * 0.625f,
            Application.BASE_WIDTH * 0.2f, 100,
            "Retry",
            new Vector2(0.5f, 0.5f));
        ContinueButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.37f, Application.BASE_HEIGHT * 0.625f,
            Application.BASE_WIDTH * 0.2f, 100,
            "Continue",
            new Vector2(0.5f, 0.5f));
        ToMainMenuButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.62f, Application.BASE_HEIGHT * 0.625f,
            Application.BASE_WIDTH * 0.2f, 100,
            "To Menu",
            new Vector2(0.5f, 0.5f));
        GameOverPanel = new GuiNPatchPanel("0.5 0.4 1100px 700px", "panel", new Vector2(0.25f, 0.25f));
        MenuPanel = new GuiNPatchPanel("0.5 0.5 1100px 500px", "panel", new Vector2(0.25f, 0.25f));
        MenuLabel = new GuiLabel("0.5 0.45 1100px 250px", "Paused", new Vector2(0.5f, 0.5f));
        MenuLabel.Color = ResourceManager.ColorLoader.Get("font_dark");

        MenuButton = new GuiTextButton("1 1 100px 50px", "Menu", Vector2.One);

        Random = new Random();
        Splats = new();

        string gameModeKey = IGameMode.GameModeTypes.Where(k => k.Value == Game.GetType()).Select(k => k.Key).Single();
        if (Application.Settings.GetTutorialEnabled(gameModeKey)) {

            if (gameModeKey != null)
                Tutorial = new TutorialDisplay(gameModeKey);
        }
    }

    /// <summary>
    /// Called when the scene is loaded. Override this method to provide custom scene initialization logic and to load resources.
    /// </summary>
    internal override void Load() {
        Input.RegisterHotkey("open_menu", KeyboardKey.KEY_ESCAPE);

        LoadAllGuiElements();

        Game.Load();
        Controller.Load();

        // Loads all the blob textures

        foreach (BlobData blobType in Game.Blobs.Values) {
            ResourceManager.TextureLoader.Load($"{blobType.Name}");

            if (ResourceManager.TextureLoader.ResourceExists($"{blobType.Name}_shadow"))
                ResourceManager.TextureLoader.Load($"{blobType.Name}_shadow");

            if (ResourceManager.TextureLoader.ResourceExists($"{blobType.Name}_splat"))
                ResourceManager.TextureLoader.Load($"{blobType.Name}_splat");

            if (ResourceManager.SoundLoader.ResourceExists($"{blobType.Name}_created"))
                ResourceManager.SoundLoader.Load($"{blobType.Name}_created");

            if (ResourceManager.SoundLoader.ResourceExists($"{blobType.Name}_destroyed"))
                ResourceManager.SoundLoader.Load($"{blobType.Name}_destroyed");
        }

        RankupArrowTexture = ResourceManager.TextureLoader.Get("rankup_arrow");
        ArenaBackgroundTexture = ResourceManager.TextureLoader.Get("arena_bg");
        ArenaBoxTexture = ResourceManager.TextureLoader.Get("arena_box");
        MarkerTexture = ResourceManager.TextureLoader.Get("marker");
        DropperTexture = ResourceManager.TextureLoader.Get("dropper");
        CurrentScoreBackgroundTexture = ResourceManager.TextureLoader.Get("score_bg");
        CurrentScoreBitmapFont = ResourceManager.TextureAtlasLoader.Get("score_spritefont");
        ScoresBackgroundTexture = ResourceManager.TextureLoader.Get("scores_bg");
        GameOverTexture = ResourceManager.TextureLoader.Get("game_over");

        Tutorial?.Load();
        IsMenuOpen = false;

        ResourceManager.WaitForLoading();
    }

    /// <summary>
    /// Called every frame to update the scene's state. 
    /// </summary>
    /// <param name="dT">The delta time since the last frame, typically used for frame-rate independent updates.</param>
    internal override void Update(float dT) {
        if (Controller is SocketController)
            Tutorial = null;

        if (IsTutorialEnabled) {
            Tutorial!.Update(dT);

            if (Tutorial.IsFinished) {
                string gameModeKey = IGameMode.GameModeTypes.Where(k => k.Value == Game.GetType()).Select(k => k.Key).Single();
                Application.Settings.SetTutorialEnabled(gameModeKey, false);
            }
        }

        if (!IsTutorialEnabled && Input.IsHotkeyActive("open_menu"))
            IsMenuOpen = !IsMenuOpen;

        if (!IsPaused) {
            Game.Update(dT);
            Controller.Update(dT, Game);

            if (Game.CanSpawnBlob && Controller.SpawnBlob(Game, out float t)) {
                t = Math.Clamp(t, 0, 1);
                Game.TrySpawnBlob(t);
            } else if (Controller.HoldBlob()) {
                Game.HoldBlob();
            }
        }

    }

    /// <summary>
    /// Called every frame to draw the scene. Override this method to provide custom scene rendering logic.
    /// </summary>
    internal override void Draw(float dT) {
        RlGl.rlPushMatrix();
        DrawScoreboard();
        DrawRankupChart();

        RlGl.rlTranslatef(ARENA_OFFSET_X, ARENA_OFFSET_Y, 0);

        DrawArenaBackground();

        Game.GameObjects.Enumerate(item => item.Draw());
        DrawSplats(dT);

        float t = Tutorial != null && !Tutorial.IsFinished ? 0.5f : Math.Clamp(Controller.GetCurrentT(), 0, 1);
        float indicatorOffset = DROP_INDICATOR_WIDTH / 2f + 1;
        float x = -IGameMode.ARENA_WIDTH / 2f + indicatorOffset + t * (IGameMode.ARENA_WIDTH - 2 * indicatorOffset);

        if (Game.IsGameOver || IsMenuOpen)
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

        Tutorial?.Draw(dT);

        if (Game.IsGameOver) {
            IsMenuOpen = false;
            DrawGameOverScreen();
        }

        if (!IsTutorialEnabled && !Game.IsGameOver && !IsMenuOpen)
            DrawMenuButton();

        if (IsMenuOpen)
            DrawMenu();
    }

    /// <summary>
    /// Called when the scene is about to be unloaded or replaced by another scene. Override this method to provide custom cleanup or deinitialization logic and to unload resources.
    /// </summary>
    internal override void Unload() {
        Input.UnregisterHotkey("open_menu");

        GameManager.Scoreboard.AddScore(Game, Game.Score);
        Controller.Close();
        // TODO unload NOT NEEDED resources
    }

    /// <summary>
    /// Called when two blobs combine.
    /// </summary>
    /// <param name="newType">The type of newly created blob.</param>
    private void Game_OnBlobsCombined(IGameMode sender, Vector2 position, int newType) {
        string key = $"{Game.Blobs[newType].Name}_created";
        if (ResourceManager.SoundLoader.ResourceExists(key)) {
            if (!ResourceManager.SoundLoader.IsLoaded(key))
                ResourceManager.SoundLoader.Load(key);
            else
                AudioManager.PlaySound(key);
        } else {
            AudioManager.PlaySound("piece_combination");
        }
    }

    private void Game_OnBlobDestroyed(IGameMode sender, Vector2 position, int type) {
        string soundKey = $"{Game.Blobs[type].Name}_destroyed";
        if (ResourceManager.SoundLoader.ResourceExists(soundKey)) {
            if (!ResourceManager.SoundLoader.IsLoaded(soundKey)) {
                ResourceManager.SoundLoader.Load(soundKey);
                ResourceManager.SoundLoader.Get(soundKey).WaitForLoad();
            }
            AudioManager.PlaySound(soundKey);
        }

        string splatKey = $"{Game.Blobs[type].Name}_splat";
        if (ResourceManager.TextureLoader.ResourceExists(splatKey)) {
            if (!ResourceManager.TextureLoader.IsLoaded(splatKey)) {
                ResourceManager.TextureLoader.Load(splatKey);
                ResourceManager.TextureLoader.Get(splatKey).WaitForLoad();
            }

            TextureResource tex = ResourceManager.TextureLoader.Get(splatKey);
            Splats[Guid.NewGuid()] = (tex, 1f, Random.NextSingle() * MathF.Tau, position);
        }
    }

    private void Game_OnGameOver(IGameMode sender) {
        if (sender.Score > GameManager.Scoreboard.GetGlobalHighscore(sender))
            AudioManager.PlaySound("new_highscore");
        else
            AudioManager.PlaySound("game_loss");
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
