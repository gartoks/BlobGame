using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using System.Numerics;

namespace BlobGame.Game.Scenes;
internal class CreditsScene : Scene {
    private GuiTextButton BackButton { get; }
    private GuiPanel BackgroundPanel { get; }

    private GuiDynamicLabel ProgrammingCaptionLabel { get; }
    private GuiDynamicLabel ArtCaptionLabel { get; }
    private GuiDynamicLabel MusicCaptionLabel { get; }
    private GuiDynamicLabel SoundCaptionLabel { get; }
    private GuiDynamicLabel OtherCaptionLabel { get; }

    private GuiDynamicLabel ProgrammersLabel { get; }
    private GuiDynamicLabel ArtistsLabel { get; }
    private GuiDynamicLabel MusiciansLabel { get; }
    private GuiDynamicLabel SoundDesignersLabel { get; }
    private GuiDynamicLabel OthersLabel { get; }

    private bool IsLoaded { get; set; }

    public CreditsScene() {
        ResourceManager.TextLoader.Load("credits");

        BackButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Back",
            new Vector2(0, 1));

        BackgroundPanel = new GuiPanel("0.05 0.05 0.9 0.8", "panel", new Vector2(0, 0));

        const float CAPTION_X_OFFSET = 0.15f;
        const float LABEL_X_OFFSET = 0.2f;

        float yOffset = 0.1f;
        ProgrammingCaptionLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * CAPTION_X_OFFSET, Application.BASE_HEIGHT * yOffset,
            "Programming",
            80f,
            new Vector2(0, 0));
        ProgrammingCaptionLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        yOffset += 0.185f;

        ArtCaptionLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * CAPTION_X_OFFSET, Application.BASE_HEIGHT * yOffset,
            "Art & UI",
            80f,
            new Vector2(0, 0));
        ArtCaptionLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        yOffset += 0.185f;

        MusicCaptionLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * CAPTION_X_OFFSET, Application.BASE_HEIGHT * yOffset,
            "Music",
            80f,
            new Vector2(0, 0));
        MusicCaptionLabel.Color = ResourceManager.ColorLoader.Get("font_dark");

        SoundCaptionLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * CAPTION_X_OFFSET * 2.5f, Application.BASE_HEIGHT * yOffset,
            "Sound",
            80f,
            new Vector2(0, 0));
        SoundCaptionLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        yOffset += 0.185f;

        OtherCaptionLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * CAPTION_X_OFFSET, Application.BASE_HEIGHT * yOffset,
            "Special Thanks",
            80f,
            new Vector2(0, 0));
        OtherCaptionLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        yOffset += 0.2f;

        yOffset = 0.2f;
        ProgrammersLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * LABEL_X_OFFSET, Application.BASE_HEIGHT * yOffset,
            string.Empty,
            50f,
            new Vector2(0, 0));
        ProgrammersLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        yOffset += 0.185f;

        ArtistsLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * LABEL_X_OFFSET, Application.BASE_HEIGHT * yOffset,
            string.Empty,
            50f,
            new Vector2(0, 0));
        ArtistsLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        yOffset += 0.185f;

        MusiciansLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * LABEL_X_OFFSET, Application.BASE_HEIGHT * yOffset,
            string.Empty,
            50f,
            new Vector2(0, 0));
        MusiciansLabel.Color = ResourceManager.ColorLoader.Get("font_dark");

        SoundDesignersLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * LABEL_X_OFFSET * 2.1f, Application.BASE_HEIGHT * yOffset,
            string.Empty,
            50f,
            new Vector2(0, 0));
        SoundDesignersLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        yOffset += 0.185f;

        OthersLabel = new GuiDynamicLabel(
            Application.BASE_WIDTH * LABEL_X_OFFSET, Application.BASE_HEIGHT * yOffset,
            string.Empty,
            50f,
            new Vector2(0, 0));
        OthersLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
    }

    internal override void Load() {
        LoadAllGuiElements();
    }

    internal override void Draw(float dT) {
        if (!IsLoaded && ResourceManager.TextLoader.Get("credits").IsLoaded) {
            TextResource credits = ResourceManager.TextLoader.Get("credits");
            ProgrammersLabel.Text = credits.Resource["programming"];
            ArtistsLabel.Text = credits.Resource["art"];
            MusiciansLabel.Text = credits.Resource["music"];
            SoundDesignersLabel.Text = credits.Resource["sound"];
            OthersLabel.Text = credits.Resource["other"];

            IsLoaded = true;
        }

        BackgroundPanel.Draw();

        ProgrammingCaptionLabel.Draw();
        ArtCaptionLabel.Draw();
        MusicCaptionLabel.Draw();
        SoundCaptionLabel.Draw();
        OtherCaptionLabel.Draw();

        ProgrammersLabel.Draw();
        ArtistsLabel.Draw();
        MusiciansLabel.Draw();
        SoundDesignersLabel.Draw();
        OthersLabel.Draw();

        BackButton.Draw();
        if (BackButton.IsClicked)
            GameManager.SetScene(new MainMenuScene());
    }
}
