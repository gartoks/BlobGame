using BlobGame.App;
using BlobGame.Game.Gui;
using BlobGame.Rendering;
using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;

namespace BlobGame.Game.Scenes;
/// <summary>
/// Scene for the main menu. First scene loaded when the game starts.
/// </summary>
internal sealed class MainMenuScene : Scene {
    private Texture TitleTexture { get; set; }
    private Texture AvatarTexture { get; set; }

    private TextScroller Scroller { get; }

    private GUIImage TitleImage { get; }
    private GuiTextButton PlayButton { get; }
    private GuiTextButton SettingsButton { get; }
    private GuiTextButton CreditsButton { get; }
    private GuiTextButton QuitButton { get; }

    public MainMenuScene() {
        Scroller = new TextScroller(5, 15, 30, 15);

        float yOffset = 0.45f;
        PlayButton = new GuiTextButton($"0.5 {yOffset} 0.25 0.125", "Play", 1, new Vector2(0.5f, 0.5f));
        yOffset += 0.15f;
        SettingsButton = new GuiTextButton($"0.5 {yOffset} 0.25 0.125", "Settings", 1, new Vector2(0.5f, 0.5f));
        yOffset += 0.15f;
        CreditsButton = new GuiTextButton($"0.5 {yOffset} 0.25 0.125", "Credits", 1, new Vector2(0.5f, 0.5f));
        yOffset += 0.15f;
        QuitButton = new GuiTextButton($"0.5 {yOffset} 0.25 0.125", "Quit", 1, new Vector2(0.5f, 0.5f));
        yOffset += 0.15f;

        TitleImage = new GUIImage(GameApplication.PROJECTION_WIDTH / 2f, GameApplication.PROJECTION_HEIGHT * 0.05f, 500, 200, 2, "title_logo", new Vector2(0.5f, 0));
    }

    internal override void Load() {
        AvatarTexture = ResourceManager.TextureLoader.GetResource("melba_avatar");

        LoadAllGuiElements();

        Scroller.Load();
    }

    internal override void Update(float dT) {
        if (PlayButton.IsClicked)
            GameManager.SetScene(new GameModeSelectionScene());
        if (SettingsButton.IsClicked)
            GameManager.SetScene(new SettingsScene());
        if (CreditsButton.IsClicked)
            GameManager.SetScene(new CreditsScene());
        if (QuitButton.IsClicked)
            GameApplication.Exit();
    }

    internal override void Render() {
        Scroller.Draw();

        // TODO: Is tmp, will fix when back
        TestDraw(-130, GameApplication.PROJECTION_HEIGHT * 1f + 130, 150, -150, 45, 10);
        TestDraw(GameApplication.PROJECTION_WIDTH * 0.7f, GameApplication.PROJECTION_HEIGHT * 1f + 170, 0, -150, 0, 14);
    }

    internal override void RenderGui() {
        base.RenderGui();

        PlayButton.Draw();
        SettingsButton.Draw();
        CreditsButton.Draw();
        QuitButton.Draw();

        float t = MathF.Sin(GameApplication.RenderGameTime * 4);
        TitleImage.Scale = 0.97f + 0.03f * t;
        TitleImage.Draw();

    }

    // TODO: Is tmp, will fix when back
    private void TestDraw(float x, float y, float dx, float dy, float rot, float time) {
        int w = AvatarTexture.Width;
        int h = AvatarTexture.Height;

        float dX = 0;
        float dY = 0;
        if (GameApplication.RenderGameTime > time) {
            float t2 = -MathF.Pow(2 * (GameApplication.RenderGameTime - time) - 1, 4) + 1;
            dX = t2 * dx;
            dY = t2 * dy;
        }

        Primitives.DrawSprite(new Vector2(x + dX, y + dY), new Vector2(x + dX + w / 4f, y + dY + h / 4f), new Vector2(1 / 2f / 4f, 1 / 2f / 4f), rot, 5, AvatarTexture, Color4.White);
    }

    internal override void Unload() {
        Scroller.Unload();
    }
}
