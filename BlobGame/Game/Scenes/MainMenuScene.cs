using BlobGame.Game.GUI;

namespace BlobGame.Game.Scenes;
/// <summary>
/// Scene for the main menu. First scene loaded when the game starts.
/// </summary>
internal sealed class MainMenuScene : Scene {

    internal override void Load() {
    }

    internal override void Update(float dT) {
    }

    internal override void Draw() {
        DrawPlayButton();
        DrawSettingsButton();
        DrawCreditsButton();
        DrawQuitButton();
    }

    internal override void Unload() {
    }

    private void DrawPlayButton() {
        float x = Application.BASE_WIDTH / 2f;
        float y = Application.BASE_HEIGHT * 0.45f;
        float w = Application.BASE_WIDTH / 4f;
        float h = Application.BASE_HEIGHT / 8f;
        bool pressed = GUITextButton.Draw(x, y, w, h, "Play");

        if (pressed)
            GameManager.SetScene(new GameScene());
    }

    private void DrawSettingsButton() {
        float w = Application.BASE_WIDTH / 4f;
        float h = Application.BASE_HEIGHT / 8f;

        float x = Application.BASE_WIDTH / 2f;
        float y = Application.BASE_HEIGHT * 0.6f;
        bool pressed = GUITextButton.Draw(x, y, w, h, "Settings");

        if (pressed)
            GameManager.SetScene(new SettingsScene());
    }

    private void DrawCreditsButton() {
        float w = Application.BASE_WIDTH / 4f;
        float h = Application.BASE_HEIGHT / 8f;

        float x = Application.BASE_WIDTH / 2f;
        float y = Application.BASE_HEIGHT * 0.75f;
        bool pressed = GUITextButton.Draw(x, y, w, h, "Credits");

        if (pressed)
            GameManager.SetScene(new CreditsScene());
    }

    private void DrawQuitButton() {
        float w = Application.BASE_WIDTH / 4f;
        float h = Application.BASE_HEIGHT / 8f;

        float x = Application.BASE_WIDTH / 2f;
        float y = Application.BASE_HEIGHT * 0.9f;
        bool pressed = GUITextButton.Draw(x, y, w, h, "Quit");

        if (pressed)
            Application.Exit();
    }
}
