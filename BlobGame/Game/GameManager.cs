using BlobGame.Game.Scenes;
using BlobGame.Game.Util;

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

    /// <summary>
    /// Initializes the game. Creates the initially loaded scene.
    /// </summary>
    internal static void Initialize() {
        //InputHandler.RegisterHotkey("w", KeyboardKey.KEY_W);

        WasSceneLoaded = false;
    }

    public static Scoreboard Scoreboard { get; }

    static GameManager() {
        Scoreboard = new Scoreboard();
    }

    /// <summary>
    /// Loads the game. Loads the initial scene.
    /// </summary>
    internal static void Load() {
        Scoreboard.Load();
        Scene = new MainMenuScene();
    }

    /// <summary>
    /// Exectues the game's update logic. Updates the currently active scene. Is executed every frame.
    /// </summary>
    /// <param name="dT"></param>
    internal static void Update(float dT) {
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
    internal static void Draw() {
        if (!WasSceneLoaded)
            return;

        Scene.Draw();
    }

    /// <summary>
    /// Unloads the game's resources.
    /// </summary>
    internal static void Unload() {
        Scene.Unload();
    }

    internal static void SetScene(Scene scene) {
        Scene.Unload();
        WasSceneLoaded = false;
        Scene = scene;
    }
}
