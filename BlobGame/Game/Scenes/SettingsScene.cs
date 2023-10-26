using BlobGame.Game.GUI;
using System.Diagnostics;
using static BlobGame.Game.GUI.GUISelector;

namespace BlobGame.Game.Scenes;
internal class SettingsScene : Scene {
    private GUITextButton BackButton { get; }
    private GUILabel ScreenModeLabel { get; }
    private GUISelector ScreenModeSelector { get; }

    //Raylib.SetWindowSize(2560, 1440);
    //Raylib.SetWindowState(ConfigFlags.FLAG_WINDOW_UNDECORATED);

    public SettingsScene() {
        BackButton = new GUITextButton(
            100, Application.BASE_HEIGHT - 150,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Back");

        SelectionElement[] elements = new SelectionElement[] {
            new SelectionElement("Fullscreen", 0),
            new SelectionElement("Windowed", 1),
            new SelectionElement("Borderless", 2),
        };
        ScreenModeSelector = new GUISelector(Application.BASE_WIDTH / 2f, 300, Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f, elements, new System.Numerics.Vector2(0.5f, 0));
    }

    internal override void Draw() {
        DrawBackButton();
    }

    private void DrawBackButton() {
        if (BackButton.Draw())
            GameManager.SetScene(new MainMenuScene());
        if (ScreenModeSelector.Draw()) {
            Debug.WriteLine(ScreenModeSelector.SelectedElement.Text);
        }
    }
}
