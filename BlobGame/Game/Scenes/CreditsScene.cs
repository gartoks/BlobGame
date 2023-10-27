using BlobGame.Game.Gui;

namespace BlobGame.Game.Scenes;
internal class CreditsScene : Scene {
    private GuiTextButton BackButton { get; }

    public CreditsScene() {
        BackButton = new GuiTextButton(
            100, Application.BASE_HEIGHT - 150,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Back");
    }

    internal override void Draw() {
        DrawBackButton();
    }

    private void DrawBackButton() {
        if (BackButton.Draw())
            GameManager.SetScene(new MainMenuScene());
    }
}
