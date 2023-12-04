using BlobGame.App;
using BlobGame.Game.Gui;
using OpenTK.Mathematics;

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
            GameApplication.PROJECTION_WIDTH * 0.05f, GameApplication.PROJECTION_HEIGHT * 0.95f,
            GameApplication.PROJECTION_WIDTH / 8f, GameApplication.PROJECTION_HEIGHT / 16f,
            "Back",
            2,
            new Vector2(0, 1));

        BackgroundPanel = new GuiPanel("0.05 0.05 0.9 0.8", 0, new Vector2(0, 0));

        float yOffset = 0.1f;
        ProgrammingCaptionLabel = new GuiDynamicLabel(
            GameApplication.PROJECTION_WIDTH * 0.1f, GameApplication.PROJECTION_HEIGHT * yOffset,
            "Programming",
            80f,
            2,
            new Vector2(0, 0));
        yOffset += 0.185f;

        ArtCaptionLabel = new GuiDynamicLabel(
            GameApplication.PROJECTION_WIDTH * 0.1f, GameApplication.PROJECTION_HEIGHT * yOffset,
            "Art",
            80f,
            2,
            new Vector2(0, 0));
        yOffset += 0.185f;

        MusicCaptionLabel = new GuiDynamicLabel(
            GameApplication.PROJECTION_WIDTH * 0.1f, GameApplication.PROJECTION_HEIGHT * yOffset,
            "Music",
            80f,
            2,
            new Vector2(0, 0));
        yOffset += 0.185f;

        OtherCaptionLabel = new GuiDynamicLabel(
            GameApplication.PROJECTION_WIDTH * 0.1f, GameApplication.PROJECTION_HEIGHT * yOffset,
            "Special Thanks",
            80f,
            2,
            new Vector2(0, 0));
        yOffset += 0.2f;

        yOffset = 0.2f;
        ProgrammersLabel = new GuiDynamicLabel(
            GameApplication.PROJECTION_WIDTH * 0.15f, GameApplication.PROJECTION_HEIGHT * yOffset,
            "gartoks\t\t\t\t\tRobotino",
            50f,
            2,
            new Vector2(0, 0));
        yOffset += 0.185f;

        ArtistsLabel = new GuiDynamicLabel(
            GameApplication.PROJECTION_WIDTH * 0.15f, GameApplication.PROJECTION_HEIGHT * yOffset,
            "Pixl\t\t\t\t\tTroobs\t\t\t\t\t_neuroFumo",
            50f,
            2,
            new Vector2(0, 0));
        yOffset += 0.185f;

        MusiciansLabel = new GuiDynamicLabel(
            GameApplication.PROJECTION_WIDTH * 0.15f, GameApplication.PROJECTION_HEIGHT * yOffset,
            "Wiggle\t\t\t\t\tFibi",
            50f,
            2,
            new Vector2(0, 0));
        yOffset += 0.185f;

        OthersLabel = new GuiDynamicLabel(
            GameApplication.PROJECTION_WIDTH * 0.15f, GameApplication.PROJECTION_HEIGHT * yOffset,
            "The Neuro Sama Discord Server <3",
            50f,
            2,
            new Vector2(0, 0));
        yOffset += 0.185f;
    }

    internal override void Load() {
        LoadAllGuiElements();
    }

    internal override void Render() {
        BackgroundPanel.Draw();

        ProgrammingCaptionLabel.Draw();
        ArtCaptionLabel.Draw();
        MusicCaptionLabel.Draw();
        OtherCaptionLabel.Draw();

        ProgrammersLabel.Draw();
        ArtistsLabel.Draw();
        MusiciansLabel.Draw();
        OthersLabel.Draw();

        BackButton.Draw();
        if (BackButton.IsClicked)
            GameManager.SetScene(new MainMenuScene());
    }
}
