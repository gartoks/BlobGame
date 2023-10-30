using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using System.Numerics;

namespace BlobGame.Game.Scenes;
internal class CreditsScene : Scene {
    private GuiTextButton BackButton { get; }
    private GuiPanel BackgroundPanel { get; }

    private GuiDynamicLabel ProgrammingCaptionLabel { get; }
    private GuiDynamicLabel ArtCaptionLabel { get; }
    private GuiDynamicLabel MusicCaptionLabel { get; }
    private GuiDynamicLabel OtherCaptionLabel { get; }

    private GuiDynamicLabel ProgrammersLabel { get; }
    private GuiDynamicLabel ArtistsLabel { get; }
    private GuiDynamicLabel MusiciansLabel { get; }
    private GuiDynamicLabel OthersLabel { get; }

    public CreditsScene() {
        BackButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Back",
            new Vector2(0, 1));

        BackgroundPanel = new GuiPanel(
                    Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.05f,
                    Application.BASE_WIDTH * 0.9f, Application.BASE_HEIGHT * 0.8f,
                    ResourceManager.GetColor("light_accent"),
                    new Vector2(0, 0));

        float yOffset = 0.1f;
        ProgrammingCaptionLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * 0.1f, Application.BASE_HEIGHT * yOffset,
            "Programming",
            80f,
            new Vector2(0, 0));
        yOffset += 0.185f;

        ArtCaptionLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * 0.1f, Application.BASE_HEIGHT * yOffset,
            "Art",
            80f,
            new Vector2(0, 0));
        yOffset += 0.185f;

        MusicCaptionLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * 0.1f, Application.BASE_HEIGHT * yOffset,
            "Music",
            80f,
            new Vector2(0, 0));
        yOffset += 0.185f;

        OtherCaptionLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * 0.1f, Application.BASE_HEIGHT * yOffset,
            "Special Thanks",
            80f,
            new Vector2(0, 0));
        yOffset += 0.2f;

        yOffset = 0.2f;
        ProgrammersLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * 0.15f, Application.BASE_HEIGHT * yOffset,
            "gartoks\t\t\t\t\tRobotino",
            50f,
            new Vector2(0, 0));
        yOffset += 0.185f;

        ArtistsLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * 0.15f, Application.BASE_HEIGHT * yOffset,
            "Pixl\t\t\t\t\tTroobs\t\t\t\t\t_neuroFumo",
            50f,
            new Vector2(0, 0));
        yOffset += 0.185f;

        MusiciansLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * 0.15f, Application.BASE_HEIGHT * yOffset,
            "Wiggle\t\t\t\t\tFibi",
            50f,
            new Vector2(0, 0));
        yOffset += 0.185f;

        OthersLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * 0.15f, Application.BASE_HEIGHT * yOffset,
            "The Neuro Sama Discord Server <3",
            50f,
            new Vector2(0, 0));
        yOffset += 0.185f;
    }

    internal override void Load() {
    }

    internal override void Draw() {
        BackgroundPanel.Draw();

        ProgrammingCaptionLabel.Draw();
        ArtCaptionLabel.Draw();
        MusicCaptionLabel.Draw();
        OtherCaptionLabel.Draw();

        ProgrammersLabel.Draw();
        ArtistsLabel.Draw();
        MusiciansLabel.Draw();
        OthersLabel.Draw();

        if (BackButton.Draw())
            GameManager.SetScene(new MainMenuScene());
    }
}
