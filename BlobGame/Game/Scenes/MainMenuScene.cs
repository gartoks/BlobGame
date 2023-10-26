using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Scenes;
/// <summary>
/// Scene for the main menu. First scene loaded when the game starts.
/// </summary>
internal sealed class MainMenuScene : Scene {

    private TextureResource TitleTexture { get; set; }
    private TextureResource StrawberryTexture { get; set; }

    private GUIImage TitleImage { get; }
    private GUITextButton PlayButton { get; }
    private GUITextButton SettingsButton { get; }
    private GUITextButton CreditsButton { get; }
    private GUITextButton QuitButton { get; }

    private (Vector2 pos, Vector2 vel, float rot)[] Strawberries { get; }

    private float Time { get; set; }

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

        Random rng = new Random();
        Strawberries = new (Vector2 pos, Vector2 vel, float rot)[80];
        float cX = Application.BASE_WIDTH / 2f;
        float cY = Application.BASE_HEIGHT / 2f;
        for (int i = 0; i < Strawberries.Length; i++) {
            float angle = rng.NextSingle() * MathF.Tau;
            float r = 1000 + 1500 * rng.NextSingle();
            float pX = MathF.Cos(angle) * r + cX;
            float pY = MathF.Sin(angle) * r + cY;

            float tX = MathF.Cos(rng.NextSingle() * MathF.Tau) * Application.BASE_HEIGHT * 0.4f + cX;
            float tY = MathF.Sin(rng.NextSingle() * MathF.Tau) * Application.BASE_HEIGHT * 0.4f + cY;

            float vX = tX - pX;
            float vY = tY - pY;
            float v = (25 + rng.NextSingle() * 25) / MathF.Sqrt((vX * vX) + (vY * vY));

            Strawberries[i] = (
                new Vector2(pX, pY),
                new Vector2(vX * v, vY * v),
                rng.NextSingle() * MathF.Tau);
        }

        Time = 0;

    }

    internal override void Load() {
        TitleTexture = ResourceManager.GetTexture("title_logo");
        StrawberryTexture = ResourceManager.GetTexture($"1");

        TitleImage.Texture = TitleTexture;
    }

    internal override void Update(float dT) {
        Time += dT;
    }

    internal override void Draw() {

        float sW = StrawberryTexture.Resource.width;
        float sH = StrawberryTexture.Resource.height;
        foreach ((Vector2 pos, Vector2 vel, float rot) sb in Strawberries) {
            Raylib.DrawTexturePro(
                StrawberryTexture.Resource,
                new Rectangle(0, 0, sW, sH),
                new Rectangle(sb.pos.X + sb.vel.X * Time, sb.pos.Y + sb.vel.Y * Time, sW, sH),
                new Vector2(0.5f * sW, 0.5f * sH),
                sb.rot + Time * 22.5f, Raylib.WHITE.ChangeAlpha(32));
        }

        if (PlayButton.Draw())
            GameManager.SetScene(new GameScene());
        if (SettingsButton.Draw())
            GameManager.SetScene(new SettingsScene());
        if (CreditsButton.Draw())
            GameManager.SetScene(new CreditsScene());
        if (QuitButton.Draw())
            Application.Exit();

        float t = MathF.Sin(Time * 4);
        TitleImage.Scale = 0.985f + 0.03f * t;
        TitleImage.Draw();
    }

    internal override void Unload() {
    }
}
