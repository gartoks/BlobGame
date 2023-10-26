using BlobGame.Drawing;
using BlobGame.Game.Gui;
using System.Diagnostics;
using System.Numerics;
using static BlobGame.Game.Gui.GUISelector;

namespace BlobGame.Game.Scenes;
internal class SettingsScene : Scene {
    private GUIPanel BackgroundPanel { get; }
    private GUITextButton BackButton { get; }
    private GUITextButton ApplyButton { get; }
    private GUILabel ScreenModeLabel { get; }
    private GUISelector ScreenModeSelector { get; }

    //Raylib.SetWindowSize(2560, 1440);
    //Raylib.SetWindowState(ConfigFlags.FLAG_WINDOW_UNDECORATED);

    public SettingsScene() {
        BackgroundPanel = new GUIPanel(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.05f,
            Application.BASE_WIDTH * 0.9f, Application.BASE_HEIGHT * 0.8f,
            Renderer.MELBA_LIGHT_PINK,
            new Vector2(0, 0));

        BackButton = new GUITextButton(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Back",
            new Vector2(0, 1));

        ApplyButton = new GUITextButton(
            Application.BASE_WIDTH * 0.95f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Apply",
            new Vector2(1, 1));

        ScreenModeLabel = new GUILabel(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.1f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f,
            "Screen Mode",
            new Vector2(0, 0));

        SelectionElement[] elements = new SelectionElement[] {
            new SelectionElement("Fullscreen", 0),
            new SelectionElement("Windowed", 1),
            new SelectionElement("Borderless", 2),
        };
        ScreenModeSelector = new GUISelector(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.1f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f,
            elements,
            new Vector2(0.5f, 0));
    }

    internal override void Draw() {
        DrawBackButton();
    }

    private void DrawBackButton() {
        BackgroundPanel.Draw();
        ScreenModeLabel.Draw();

        if (BackButton.Draw())
            GameManager.SetScene(new MainMenuScene());
        if (ApplyButton.Draw())
            GameManager.SetScene(new MainMenuScene());
        if (ScreenModeSelector.Draw()) {
            Debug.WriteLine(ScreenModeSelector.SelectedElement.Text);
        }
    }
}
