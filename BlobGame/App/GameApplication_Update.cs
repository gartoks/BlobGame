using BlobGame.Audio;
using BlobGame.Game;
using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using SimpleGL;

namespace BlobGame.App;
internal sealed partial class GameApplication : Application {

    public static float UpdateGameTime { get; private set; }

    public override void OnUpdateStart() {
        AudioManager.Initialize();
        Input.Initialize();
        //GUIHandler.Initialize();
        Fonts.Initialize();
        GameManager.Initialize();
        GuiManager.Initialize();

        Settings.Load();
        AudioManager.Load();
        Input.Load();
        Fonts.Load();
        GameManager.Load();
        GuiManager.Load();
    }

    public override void OnUpdate(float deltaTime) {
        UpdateGameTime += deltaTime;

        AudioManager.Update();
        Input.Update();
        GameManager.Update(deltaTime);
        GuiManager.Update(deltaTime);
    }

    public override void OnUpdateStop() {
        GameManager.Unload();
        Fonts.Unload();
        AudioManager.Unload();
    }
}
