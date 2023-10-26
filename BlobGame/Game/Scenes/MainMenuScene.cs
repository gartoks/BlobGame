using BlobGame.Drawing;
using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using System.Numerics;

namespace BlobGame.Game.Scenes;
/// <summary>
/// Scene for the main menu. First scene loaded when the game starts.
/// </summary>
internal sealed class MainMenuScene : Scene {

    private TextureResource TitleTexture { get; set; }

    private GUIImage TitleImage { get; }
    private GUITextButton PlayButton { get; }
    private GUITextButton SettingsButton { get; }
    private GUITextButton CreditsButton { get; }
    private GUITextButton QuitButton { get; }

    public MainMenuScene() {
        PlayButton = new GUITextButton(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.45f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 8f,
            "Play",
            new Vector2(0.5f, 0.5f));
        SettingsButton = new GUITextButton(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.6f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 8f,
            "Settings",
            new Vector2(0.5f, 0.5f));
        CreditsButton = new GUITextButton(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.75f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 8f,
            "Credits",
            new Vector2(0.5f, 0.5f));
        QuitButton = new GUITextButton(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.9f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 8f,
            "Quit",
            new Vector2(0.5f, 0.5f));

        TitleImage = new GUIImage(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.05f,
            1,
            //Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.2f,
            ResourceManager.DefaultTexture,
            new Vector2(0.5f, 0));
    }

    internal override void Load() {
        TitleTexture = ResourceManager.GetTexture("title_logo");

        TitleImage.Texture = TitleTexture;
    }

    internal override void Update(float dT) {
    }

    internal override void Draw() {
        if (PlayButton.Draw())
            GameManager.SetScene(new GameScene());
        if (SettingsButton.Draw())
            GameManager.SetScene(new SettingsScene());
        if (CreditsButton.Draw())
            GameManager.SetScene(new CreditsScene());
        if (QuitButton.Draw())
            Application.Exit();

        float t = MathF.Sin(Renderer.Time * 4);
        TitleImage.Scale = 0.985f + 0.03f * t;
        TitleImage.Draw();
    }

    internal override void Unload() {
    }
}
