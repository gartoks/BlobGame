using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using System.Numerics;

namespace BlobGame.Game.Scenes;
internal class ControlsScene : Scene {
    private GuiTextButton BackButton { get; }
    private GuiPanel BackgroundPanel { get; }

    private GuiDynamicLabel MouseControlsLabel { get; set; }
    private GuiDynamicLabel MouseControlsDescription { get; set; }

    private GuiDynamicLabel KeyboardControlsLabel { get; set; }
    private GuiDynamicLabel KeyboardControlsDescription { get; set; }

    private GuiDynamicLabel SocketControlsLabel { get; set; }
    private GuiDynamicLabel SocketControlsDescription { get; set; }

    public ControlsScene() {
        BackButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Back",
            new Vector2(0, 1));

        BackgroundPanel = new GuiPanel("0.05 0.05 0.9 0.8", "panel", new Vector2(0, 0));

        float y = 0.125f;
        MouseControlsLabel = new GuiDynamicLabel(Application.BASE_WIDTH * 0.125f, Application.BASE_HEIGHT * y, "Mouse", 100);
        MouseControlsLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        MouseControlsDescription = new GuiDynamicLabel(Application.BASE_WIDTH * 0.15f, Application.BASE_HEIGHT * (y + 0.1f), "Left Click: Drop Piece\nMiddle Click: Hold Piece", 70);
        MouseControlsDescription.Color = ResourceManager.ColorLoader.Get("font_dark");

        KeyboardControlsLabel = new GuiDynamicLabel(Application.BASE_WIDTH * 0.525f, Application.BASE_HEIGHT * y, "Keyboard", 100);
        KeyboardControlsLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        KeyboardControlsDescription = new GuiDynamicLabel(Application.BASE_WIDTH * 0.55f, Application.BASE_HEIGHT * (y + 0.1f), "W & S: Move Dropper\nSpace: Drop Piece\nTab: Hold Piece", 70);
        KeyboardControlsDescription.Color = ResourceManager.ColorLoader.Get("font_dark");

        SocketControlsLabel = new GuiDynamicLabel(Application.BASE_WIDTH * 0.125f, Application.BASE_HEIGHT * (y + 0.3f), "Socket", 100);
        SocketControlsLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        SocketControlsDescription = new GuiDynamicLabel(Application.BASE_WIDTH * 0.15f, Application.BASE_HEIGHT * (y + 0.3f + 0.1f), "See external file\nResources/HowTo.txt", 70);
        SocketControlsDescription.Color = ResourceManager.ColorLoader.Get("font_dark");
    }

    internal override void Load() {
        LoadAllGuiElements();
    }

    internal override void Draw(float dT) {
        BackgroundPanel.Draw();

        MouseControlsLabel.Draw();
        MouseControlsDescription.Draw();
        KeyboardControlsLabel.Draw();
        KeyboardControlsDescription.Draw();
        SocketControlsLabel.Draw();
        SocketControlsDescription.Draw();

        BackButton.Draw();
        if (BackButton.IsClicked)
            GameManager.SetScene(new MainMenuScene());
    }
}
