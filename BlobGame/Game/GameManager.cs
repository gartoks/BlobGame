using BlobGame.Audio;
using BlobGame.Drawing;
using BlobGame.Game.Gui;
using BlobGame.Game.Scenes;
using BlobGame.Game.Util;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game;
/// <summary>
/// Static class to control the main game. Handles the game's scenes.
/// </summary>
public static class GameManager {
    /// <summary>
    /// The currently active scene.
    /// </summary>
    private static Scene Scene { get; set; }
    /// <summary>
    /// Flag indicating if the scene was loaded after setting a new scene.
    /// </summary>
    private static bool WasSceneLoaded { get; set; }

    public static Scoreboard Scoreboard { get; }

    /// <summary>
    /// Tubmler to draw cute little strawberries in the background.
    /// </summary>
    private static BackgroundTumbler Tumbler { get; }

    private static IReadOnlyList<MusicResource> Music { get; set; }
    private static bool WasMusicQueued { get; set; }

    static GameManager() {
        Scoreboard = new Scoreboard();
        Tumbler = new BackgroundTumbler(60);
    }

    /// <summary>
    /// Initializes the game. Creates the initially loaded scene.
    /// </summary>
    internal static void Initialize() {
        //InputHandler.RegisterHotkey("w", KeyboardKey.KEY_W);

        WasSceneLoaded = false;
        WasMusicQueued = false;

        GuiManager.Initialize();
    }

    /// <summary>
    /// Loads the game. Loads the initial scene.
    /// </summary>
    internal static void Load() {
        Scoreboard.Load();

        Tumbler.Load();

        Music = new MusicResource[] {
            ResourceManager.GetMusic("crossinglike"),
            ResourceManager.GetMusic("Melba_1"),
            ResourceManager.GetMusic("Melba_2"),
            ResourceManager.GetMusic("Melba_3"),
            ResourceManager.GetMusic("Melba_s_Toasty_Game"),
            ResourceManager.GetMusic("On_the_Surface"),
            ResourceManager.GetMusic("synthyupdated"),
        };

        GuiManager.Load();

        Scene = new MainMenuScene();
    }

    /// <summary>
    /// Exectues the game's update logic. Updates the currently active scene. Is executed every frame.
    /// </summary>
    /// <param name="dT"></param>
    internal static void Update(float dT) {
        if (Music.All(m => !AudioManager.IsMusicPlaying(m.Key))) {
            if (WasMusicQueued)
                return;

            Random rng = new Random();
            AudioManager.PlayMusic(Music[rng.Next(Music.Count)].Key);
            WasMusicQueued = true;
        } else {
            WasMusicQueued = false;
        }

        GuiManager.Update(dT);

        // The scene is loaded in the update method to ensure scene drawing doesn't access unloaded resources.
        if (!WasSceneLoaded) {
            Scene.Load();
            WasSceneLoaded = true;
        } else
            Scene.Update(dT);
    }

    /// <summary>
    /// Draws the game. Is executed every frame.
    /// </summary>
    internal static void Draw(float dT) {
        DrawBackground();
        Tumbler.Draw(dT);

        if (!WasSceneLoaded)
            return;

        Scene.Draw();
    }

    /// <summary>
    /// Unloads the game's resources.
    /// </summary>
    internal static void Unload() {
        Scene.Unload();
        GuiManager.Unload();
    }

    internal static void SetScene(Scene scene) {
        Scene.Unload();
        GuiManager.ResetElements();
        WasSceneLoaded = false;
        Scene = scene;
    }

    private static void DrawBackground() {
        const float ANGLE = -12.5f;
        Color elementColor = ResourceManager.GetColor("light_accent").Resource.ChangeAlpha(64);
        //Color elementColor = new Color(255, 255, 255, 64);

        Raylib.DrawRectanglePro(
            new Rectangle(-100, 287.5f, 2500, 100),
            new Vector2(), ANGLE, elementColor);

        Raylib.DrawRectanglePro(
            new Rectangle(-100, Application.BASE_HEIGHT * 0.80f, 2500, 25),
            new Vector2(), ANGLE, elementColor);

        Raylib.DrawRectanglePro(
            new Rectangle(-100, Application.BASE_HEIGHT * 0.85f, 2500, 200),
            new Vector2(), ANGLE, elementColor);
    }
}
