using BlobGame.Game.Scenes;

namespace BlobGame.Game;
/// <summary>
/// Static class to control the main game. Handles the game's scenes.
/// </summary>
public static class Game {
    /// <summary>
    /// The currently active scene.
    /// </summary>
    private static Scene Scene { get; set; }

    /// <summary>
    /// Initializes the game. Creates the initially loaded scene.
    /// </summary>
    internal static void Initialize() {
        //InputHandler.RegisterHotkey("w", KeyboardKey.KEY_W);

        Scene = new GameScene();
    }

    /// <summary>
    /// Loads the game. Loads the initial scene.
    /// </summary>
    internal static void Load() {
        Scene.Load();
    }

    /// <summary>
    /// Exectues the game's update logic. Updates the currently active scene. Is executed every frame.
    /// </summary>
    /// <param name="dT"></param>
    internal static void Update(float dT) {
        Scene.Update(dT);
    }

    /// <summary>
    /// Draws the game. Is executed every frame.
    /// </summary>
    public static void Draw() {
        Scene.Draw();
    }
}
