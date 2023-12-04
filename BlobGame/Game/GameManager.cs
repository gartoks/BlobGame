using BlobGame.App;
using BlobGame.Audio;
using BlobGame.Game.Gui;
using BlobGame.Game.Scenes;
using BlobGame.Game.Util;
using BlobGame.Rendering;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Util.Math;

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
    /// The next scene to be loaded
    /// </summary>
    private static Scene? NextScene { get; set; }

    public static Scoreboard Scoreboard { get; }

    /// <summary>
    /// Tubmler to draw cute little strawberries in the background.
    /// </summary>
    private static BackgroundTumbler Tumbler { get; }

    private static IReadOnlyList<(string key, Music audio)> Music { get; set; }
    private static bool WasMusicQueued { get; set; }

    private static bool WereResourcesLoaded { get; set; }

    static GameManager() {
        Scoreboard = new Scoreboard();
        Tumbler = new BackgroundTumbler(60);

        WereResourcesLoaded = false;
    }

    /// <summary>
    /// Initializes the game. Creates the initially loaded scene.
    /// </summary>
    internal static void Initialize() {
        //InputHandler.RegisterHotkey("w", KeyboardKey.KEY_W);

        WasMusicQueued = false;
    }

    /// <summary>
    /// Loads the game. Loads the initial scene.
    /// </summary>
    internal static void Load() {
        NextScene = new GameLoadingScene();
    }

    internal static void LoadResources() {
        Scoreboard.Load();

        Tumbler.Load();

        Music = new (string key, Music audio)[] {
            ("crossinglike", ResourceManager.MusicLoader.GetResource("crossinglike")),
            ("Melba_1", ResourceManager.MusicLoader.GetResource("Melba_1")),
            ("Melba_2", ResourceManager.MusicLoader.GetResource("Melba_2")),
            ("Melba_3", ResourceManager.MusicLoader.GetResource("Melba_3")),
            ("Melba_s_Toasty_Game", ResourceManager.MusicLoader.GetResource("Melba_s_Toasty_Game")),
            ("On_the_Surface", ResourceManager.MusicLoader.GetResource("On_the_Surface")),
            ("synthyupdated", ResourceManager.MusicLoader.GetResource("synthyupdated")),
        };

        WereResourcesLoaded = true;
    }

    /// <summary>
    /// Exectues the game's update logic. Updates the currently active scene. Is executed every frame.
    /// </summary>
    /// <param name="dT"></param>
    internal static void Update(float dT) {
        if (WereResourcesLoaded) {
            if (Music.All(m => !AudioManager.IsMusicPlaying(m.key))) {
                if (WasMusicQueued)
                    return;

                Random rng = new Random();
                AudioManager.PlayMusic(Music[rng.Next(Music.Count)].key);
                WasMusicQueued = true;
            } else {
                WasMusicQueued = false;
            }
        }

        // The scene is loaded in the update method to ensure scene drawing doesn't access unloaded resources.
        if (NextScene != null) {
            Scene?.Unload();
            GuiManager.ResetElements();
            Scene = NextScene;
            Scene.Load();
            NextScene = null;
        } else
            Scene.Update(dT);
    }

    /// <summary>
    /// Draws the game. Is executed every frame.
    /// </summary>
    internal static void Render(float dT) {
        if (WereResourcesLoaded) {
            DrawBackground();
            Tumbler.Draw(dT);
        }

        if (NextScene != null)
            return;

        Scene.Render();
    }

    internal static void RenderGui(float dT) {
        if (NextScene != null)
            return;

        Scene?.RenderGui();
    }

    /// <summary>
    /// Unloads the game's resources.
    /// </summary>
    internal static void Unload() {
        Scene.Unload();
        GuiManager.Unload();
    }

    internal static void SetScene(Scene scene) {
        NextScene = scene;
    }

    private static void DrawBackground() {
        float ANGLE = -12.5f.ToRad();
        Color4 elementColor = ResourceManager.ColorLoader.GetResource("light_accent").ChangeAlpha(64);
        //Color elementColor = new Color(255, 255, 255, 64);

        Primitives.DrawRectangle(
            new Vector2(-100, 287.5f),
            new Vector2(2500, 100),
            new Vector2(), ANGLE, 0, elementColor);

        Primitives.DrawRectangle(
            new Vector2(-100, GameApplication.PROJECTION_HEIGHT * 0.80f),
            new Vector2(2500, 25),
            new Vector2(), ANGLE, 0, elementColor);

        Primitives.DrawRectangle(
            new Vector2(-100, GameApplication.PROJECTION_HEIGHT * 0.85f),
            new Vector2(2500, 200),
            new Vector2(), ANGLE, 0, elementColor);
    }
}
