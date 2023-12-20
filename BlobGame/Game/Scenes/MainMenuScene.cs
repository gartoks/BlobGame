using BlobGame.App;
using BlobGame.Drawing;
using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using System.Numerics;

namespace BlobGame.Game.Scenes;
/// <summary>
/// Scene for the main menu. First scene loaded when the game starts.
/// </summary>
internal sealed class MainMenuScene : Scene {
    private TextureResource TitleTexture { get; set; }

    private TextScroller Scroller { get; }

    private GUIImage TitleImage { get; }
    private GuiTextButton PlayButton { get; }
    private GuiTextButton SettingsButton { get; }
    private GuiTextButton ControlsButton { get; }
    private GuiTextButton CreditsButton { get; }
    private GuiTextButton QuitButton { get; }
    private GuiTextButton LoginButton { get; }
    private GuiDynamicLabel UsernameDisplay { get; }

    public MainMenuScene() {
        Scroller = new TextScroller(5, 15, 30, 15);

        float yOffset = 0.4f;
        PlayButton = new GuiTextButton(
            0.5f * Application.BASE_WIDTH, yOffset * Application.BASE_HEIGHT,
            0.25f * Application.BASE_WIDTH, 0.1f * Application.BASE_HEIGHT,
            "Play",
            new Vector2(0.5f, 0.5f));
        yOffset += 0.125f;
        SettingsButton = new GuiTextButton(
            0.5f * Application.BASE_WIDTH, yOffset * Application.BASE_HEIGHT,
            0.25f * Application.BASE_WIDTH, 0.1f * Application.BASE_HEIGHT,
            "Settings",
            new Vector2(0.5f, 0.5f));
        yOffset += 0.125f;
        ControlsButton = new GuiTextButton(
            0.5f * Application.BASE_WIDTH, yOffset * Application.BASE_HEIGHT,
            0.25f * Application.BASE_WIDTH, 0.1f * Application.BASE_HEIGHT,
            "Controls",
            new Vector2(0.5f, 0.5f));
        yOffset += 0.125f;
        CreditsButton = new GuiTextButton(
            0.5f * Application.BASE_WIDTH, yOffset * Application.BASE_HEIGHT,
            0.25f * Application.BASE_WIDTH, 0.1f * Application.BASE_HEIGHT,
            "Credits",
            new Vector2(0.5f, 0.5f));
        yOffset += 0.125f;
        QuitButton = new GuiTextButton(
            0.5f * Application.BASE_WIDTH, yOffset * Application.BASE_HEIGHT,
            0.25f * Application.BASE_WIDTH, 0.1f * Application.BASE_HEIGHT,
            "Quit",
            new Vector2(0.5f, 0.5f));
        yOffset += 0.125f;

        TitleImage = new GUIImage(
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT * 0.05f,
            0.5f,
            ResourceManager.TextureLoader.Fallback,
            new Vector2(0.5f, 0));

        UsernameDisplay = new GuiDynamicLabel(0, Application.BASE_HEIGHT - 50, "Checking user...", 50);
        UsernameDisplay.Color = ResourceManager.ColorLoader.Get("font_dark");

        LoginButton = new GuiTextButton(0, Application.BASE_HEIGHT - 100, 300, 50, "Sign in with Discord");


        DiscordAuth.UpdateUserInfo().ContinueWith((Task _) => {
            UsernameDisplay.Text = DiscordAuth.Username;
            if (DiscordAuth.IsSignedIn) {
                LoginButton.Label.Text = "Sign out";
            }
        });
    }

    internal override void Load() {
        TitleTexture = ResourceManager.TextureLoader.Get("title_logo");

        // Preload so the game doesnt lag on startup so much
        ResourceManager.TextureLoader.Load("avatar_idle");
        ResourceManager.TextureLoader.Load("avatar_blink_0");
        ResourceManager.TextureLoader.Load("avatar_blink_1");
        ResourceManager.TextureLoader.Load("avatar_talk_0");
        ResourceManager.TextureLoader.Load("avatar_talk_1");
        ResourceManager.TextureLoader.Load("avatar_talk_2");
        ResourceManager.TextureLoader.Load("avatar_overlay_0");
        ResourceManager.TextureLoader.Load("avatar_overlay_1");
        ResourceManager.TextureLoader.Load("avatar_overlay_2");

        LoadAllGuiElements();

        Scroller.Load();

        TitleImage.Texture = TitleTexture;
    }

    internal override void Update(float dT) {
    }

    internal override void Draw(float dT) {
        Scroller.Draw();

        PlayButton.Draw();
        SettingsButton.Draw();
        ControlsButton.Draw();
        CreditsButton.Draw();
        QuitButton.Draw();
        LoginButton.Draw();

        if (PlayButton.IsClicked)
            GameManager.SetScene(new GameModeSelectionScene());
        if (SettingsButton.IsClicked)
            GameManager.SetScene(new SettingsScene());
        if (ControlsButton.IsClicked)
            GameManager.SetScene(new ControlsScene());
        if (CreditsButton.IsClicked)
            GameManager.SetScene(new CreditsScene());
        if (QuitButton.IsClicked)
            Application.Exit();
        if (LoginButton.IsClicked) {
            UsernameDisplay.Text = "Loading...";

            Task task;
            if (DiscordAuth.IsSignedIn)
                task = DiscordAuth.SignOut();
            else
                task = DiscordAuth.SignIn();

            task.ContinueWith(async (Task _) => {
                await DiscordAuth.UpdateUserInfo();
                Application.Settings.DiscordTokensChanged();
                UsernameDisplay.Text = DiscordAuth.Username;

                LoginButton.Label.Text = DiscordAuth.IsSignedIn ? "Sign out" : "Sign in with Discord";
            });
        }

        float t = MathF.Sin(Renderer.Time * 4);
        TitleImage.Scale = 0.485f + 0.03f * t;
        TitleImage.Draw();

        UsernameDisplay.Draw();

        /*// TODO: Is tmp, will fix when back
        TestDraw(-130, Application.BASE_HEIGHT * 1f + 130, 150, -150, 45, 10);
        TestDraw(Application.BASE_WIDTH * 0.7f, Application.BASE_HEIGHT * 1f + 170, 0, -150, 0, 14);*/
    }
    /*
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
        }*/

    internal override void Unload() {
        Scroller.Unload();
    }
}
