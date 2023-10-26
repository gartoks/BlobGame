using BlobGame.Game.Scenes;

namespace BlobGame.Game;
/// <summary>
/// Static class to control the main game. Handles the game's scenes.
/// </summary>
public static class GameManager {
    /// <summary>
    /// The currently active scene.
    /// </summary>
    private static Scene Scene { get; set; }
    private static bool WasSceneLoaded { get; set; }

    /// <summary>
    /// Initializes the game. Creates the initially loaded scene.
    /// </summary>
    internal static void Initialize() {
        //InputHandler.RegisterHotkey("w", KeyboardKey.KEY_W);

        WasSceneLoaded = false;
    }

    /// <summary>
    /// Loads the game. Loads the initial scene.
    /// </summary>
    internal static void Load() {
        Scene = new MainMenuScene();
    }

    /// <summary>
    /// Exectues the game's update logic. Updates the currently active scene. Is executed every frame.
    /// </summary>
    /// <param name="dT"></param>
    internal static void Update(float dT) {
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
