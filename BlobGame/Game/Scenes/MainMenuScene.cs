using BlobGame.Drawing;
using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Scenes;
/// <summary>
/// Scene for the main menu. First scene loaded when the game starts.
/// </summary>
internal sealed class MainMenuScene : Scene {
    private TextureResource TitleTexture { get; set; }
    private TextureResource AvatarTexture { get; set; }

    private TextScroller Scroller { get; }

    private GUIImage TitleImage { get; }
    private GuiTextButton PlayButton { get; }
    private GuiTextButton SettingsButton { get; }
    private GuiTextButton CreditsButton { get; }
    private GuiTextButton QuitButton { get; }

    public MainMenuScene() {
        Scroller = new TextScroller(15, 45, 15);

        PlayButton = new GuiTextButton(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.45f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 8f,
            "Play",
            new Vector2(0.5f, 0.5f));
        SettingsButton = new GuiTextButton(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.6f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 8f,
            "Settings",
            new Vector2(0.5f, 0.5f));
        CreditsButton = new GuiTextButton(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.75f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 8f,
            "Credits",
            new Vector2(0.5f, 0.5f));
        QuitButton = new GuiTextButton(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.9f,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 8f,
            "Quit",
            new Vector2(0.5f, 0.5f));

        TitleImage = new GUIImage(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.05f,
            1,
            //Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.2f,
            ResourceManager.FallbackTexture,
            new Vector2(0.5f, 0));
    }

    internal override void Load() {
        TitleTexture = ResourceManager.GetTexture("title_logo");
        AvatarTexture = ResourceManager.GetTexture("melba_avatar");

        Scroller.Load();

        TitleImage.Texture = TitleTexture;
    }

    internal override void Update(float dT) {
    }

    internal override void Draw() {
        Scroller.Draw();

        if (PlayButton.Draw())
            GameManager.SetScene(new GameModeSelectionScene());
        if (SettingsButton.Draw())
            GameManager.SetScene(new SettingsScene());
        if (CreditsButton.Draw())
            GameManager.SetScene(new CreditsScene());
        if (QuitButton.Draw())
            Application.Exit();

        float t = MathF.Sin(Renderer.Time * 4);
        TitleImage.Scale = 0.985f + 0.03f * t;
        TitleImage.Draw();

        // TODO: Is tmp, will fix when back
        TestDraw(-130, Application.BASE_HEIGHT * 1f + 130, 150, -150, 45, 10);
        TestDraw(Application.BASE_WIDTH * 0.7f, Application.BASE_HEIGHT * 1f + 170, 0, -150, 0, 14);
    }

    // TODO: Is tmp, will fix when back
    private void TestDraw(float x, float y, float dx, float dy, float rot, float time) {
        int w = AvatarTexture.Resource.width;
        int h = AvatarTexture.Resource.height;

        float dX = 0;
        float dY = 0;
        if (Renderer.Time > time) {
            float t2 = -MathF.Pow(2 * (Renderer.Time - time) - 1, 4) + 1;
            dX = t2 * dx;
            dY = t2 * dy;
        }

        Raylib.DrawTexturePro(AvatarTexture.Resource, new Rectangle(0, 0, w, h), new Rectangle(x + dX, y + dY, w / 4f, h / 4f), new Vector2(w / 2f / 4f, h / 2f / 4f), rot, Raylib.WHITE);
    }

    internal override void Unload() {
        Scroller.Unload();
    }
}
