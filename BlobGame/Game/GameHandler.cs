using BlobGame.Game.Scenes;

namespace BlobGame.Game;
public static class GameHandler {

    private static Scene Scene { get; set; }

    static GameHandler() {
    }

    internal static void Initialize() {
        //InputHandler.RegisterHotkey("w", KeyboardKey.KEY_W);
        //InputHandler.RegisterHotkey("a", KeyboardKey.KEY_A);
        //InputHandler.RegisterHotkey("s", KeyboardKey.KEY_S);
        //InputHandler.RegisterHotkey("d", KeyboardKey.KEY_D);
        //InputHandler.RegisterHotkey("q", KeyboardKey.KEY_Q);
        //InputHandler.RegisterHotkey("e", KeyboardKey.KEY_E);

        Scene = new GameScene();
    }

    internal static void Load() {
        Scene.Load();
    }

    internal static void Update(float dT) {
        Scene.Update(dT);
    }

    public static void Draw() {
        Scene.Draw();
    }
}
