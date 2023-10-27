using BlobGame.App;
using BlobGame.Game.Gui;
using BlobGame.Game.Util;
using BlobGame.ResourceHandling;
using System.Numerics;
using static BlobGame.Game.Gui.GuiSelector;
using static BlobGame.Game.Util.Settings;

namespace BlobGame.Game.Scenes;
internal class SettingsScene : Scene {
    private GuiPanel BackgroundPanel { get; }

    private GuiTextButton BackButton { get; }
    private GuiTextButton ApplyButton { get; }

    private GuiLabel ScreenModeLabel { get; }
    private GuiSelector ScreenModeSelector { get; }

    private GuiLabel ResolutionLabel { get; }
    private GuiSelector ResolutionSelector { get; }

    private GuiLabel MonitorLabel { get; }
    private GuiSelector MonitorSelector { get; }
    private GuiLabel ThemeLabel { get; }
    private GuiSelector ThemeSelector { get; }

    private GuiLabel MusicVolumeLabel { get; }
    private GuiSelector MusicVolumeSelector { get; }

    private GuiLabel SoundVolumeLabel { get; }
    private GuiSelector SoundVolumeSelector { get; }

    private GuiTextButton ResetScoreButton { get; }
    private GuiTextButton ResetTutorialButton { get; }


    internal override void Load() {
    }

    public SettingsScene() {
        BackgroundPanel = new GuiPanel(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.05f,
            Application.BASE_WIDTH * 0.9f, Application.BASE_HEIGHT * 0.8f,
            ResourceManager.GetColor("light_accent"),
            new Vector2(0, 0));

        BackButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Back",
            new Vector2(0, 1));

        ApplyButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.95f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Apply",
            new Vector2(1, 1));

        float xOffset = 0.1f;
        (GuiSelector monitorSelector, GuiLabel monitorLabel) = CreateSettingsEntry(
            "Monitor", xOffset,
            Application.Settings.GetMonitors().Select(m => new SelectionElement($"{m.monitor}: {m.name}", m.monitor)).ToArray(),
            Array.FindIndex(Application.Settings.GetMonitors(), m => m.monitor == Application.Settings.GetCurrentMonitor()));
        MonitorLabel = monitorLabel;
        MonitorSelector = monitorSelector;
        //xOffset += 0.1f;

        (GuiSelector screenModeSelector, GuiLabel screenModeLabel) = CreateSettingsEntry(
            "Screen Mode", xOffset,
            Enum.GetValues<eScreenMode>().Select(sm => new SelectionElement(sm.ToString(), sm)).ToArray(),
            Array.FindIndex(Enum.GetValues<eScreenMode>(), sm => sm == Application.Settings.ScreenMode));
        ScreenModeLabel = screenModeLabel;
        ScreenModeSelector = screenModeSelector;
        xOffset += 0.1f;

        (GuiSelector resolutionSelector, GuiLabel resolutionLabel) = CreateSettingsEntry(
            "Resolution", xOffset,
            Settings.AVAILABLE_RESOLUTIONS.Select(res => new SelectionElement($"{res.w}x{res.h}", res)).ToArray(),
            Array.FindIndex<(int, int)>(Settings.AVAILABLE_RESOLUTIONS.ToArray(), r => r.Equals(Application.Settings.GetCurrentResolution())));
        ResolutionLabel = resolutionLabel;
        ResolutionSelector = resolutionSelector;
        xOffset += 0.1f;

        (GuiSelector musicVolumeSelector, GuiLabel musicVolumeLabel) = CreateSettingsEntry(
            "Music Volume", xOffset,
            Enumerable.Range(0, 11).Select(i => new SelectionElement($"{i * 10f}%", i * 10)).ToArray(),
            Application.Settings.MusicVolume / 10);
        MusicVolumeLabel = musicVolumeLabel;
        MusicVolumeSelector = musicVolumeSelector;
        xOffset += 0.1f;

        (GuiSelector soundVolumeSelector, GuiLabel soundVolumeLabel) = CreateSettingsEntry(
            "Sound Volume", xOffset,
            Enumerable.Range(0, 11).Select(i => new SelectionElement($"{i * 10f}%", i * 10)).ToArray(),
            Application.Settings.SoundVolume / 10);
        SoundVolumeLabel = soundVolumeLabel;
        SoundVolumeSelector = soundVolumeSelector;
        xOffset += 0.1f;

        SelectionElement[] availableThemes = Directory.GetFiles(Files.GetResourceFilePath("Themes"))
            .Where(file => file.EndsWith(".theme"))
            .Select(file => new SelectionElement($"{Path.GetFileNameWithoutExtension(file)}", Path.GetFileNameWithoutExtension(file))).ToArray();
        int selectedThemeIndex = Array.FindIndex(availableThemes, e => e.Element.Equals(Application.Settings.GetCurrentThemeName()));

        (GuiSelector themeSelector, GuiLabel themeLabel) = CreateSettingsEntry(
            "Theme", xOffset,
            availableThemes,
            selectedThemeIndex);
        ThemeSelector = themeSelector;
        ThemeLabel = themeLabel;
        xOffset += 0.1f;

        ResetScoreButton = new GuiTextButton(
            new Vector2(Application.BASE_WIDTH * 0.1f, Application.BASE_HEIGHT * 0.8f),
            new Vector2(Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f),
            "Reset Score",
            new Vector2(0, 1));

        ResetTutorialButton = new GuiTextButton(
            new Vector2(Application.BASE_WIDTH * 0.85f, Application.BASE_HEIGHT * 0.8f),
            new Vector2(Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f),
            "Reset Tutorial",
            new Vector2(1, 1));
    }

    internal override void Draw() {
        DrawBackButton();
    }

    private void DrawBackButton() {
        BackgroundPanel.Draw();
        //MonitorLabel.Draw(); // TODO: Currently disabled because it is not working.
        ScreenModeLabel.Draw();
        ResolutionLabel.Draw();
        MusicVolumeLabel.Draw();
        SoundVolumeLabel.Draw();
        ThemeLabel.Draw();

        if (BackButton.Draw())
            GameManager.SetScene(new MainMenuScene());
        if (ApplyButton.Draw())
            ApplySettings();

        //MonitorSelector.Draw();   // TODO: Currently disabled because it is not working.
        ScreenModeSelector.Draw();
        ResolutionSelector.Draw();
        MusicVolumeSelector.Draw();
        SoundVolumeSelector.Draw();
        ThemeSelector.Draw();

        if (ResetScoreButton.Draw())
            GameManager.Scoreboard.Reset();
        if (ResetTutorialButton.Draw())
            Application.Settings.IsTutorialEnabled = true;
    }

    private void ApplySettings() {
        int monitor = (int)MonitorSelector.SelectedElement.Element;
        eScreenMode screenMode = (eScreenMode)ScreenModeSelector.SelectedElement.Element;
        (int w, int h) resolution = ((int w, int h))ResolutionSelector.SelectedElement.Element;
        string theme = (string)ThemeSelector.SelectedElement.Element;

        if (resolution != Application.Settings.GetCurrentResolution())
            Application.Settings.SetResolution(resolution.w, resolution.h);
        if (screenMode != Application.Settings.ScreenMode)
            Application.Settings.SetScreenMode(screenMode);
        if (monitor != Application.Settings.GetCurrentMonitor())
            Application.Settings.SetMonitor(monitor);
        if (theme != Application.Settings.GetCurrentThemeName())
            Application.Settings.SetTheme(theme);

#if WINDOWS
        // This crashes on linux for some reason. Everything works without it, so it doesn't matter
        Application.Exit();
        Process.Start(Assembly.GetExecutingAssembly().Location);
#endif
        GameManager.SetScene(new SettingsScene());
    }

    private (GuiSelector, GuiLabel) CreateSettingsEntry(string name, float xOffset, SelectionElement[] selectionElements, int selectedIndex) {
        GuiLabel label = new GuiLabel(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f,
            name,
            new Vector2(0, 0));

        GuiSelector selector = new GuiSelector(
            Application.BASE_WIDTH * 0.35f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT / 16f,
            selectionElements, selectedIndex < 0 ? 0 : selectedIndex,
            new Vector2(0, 0));

        return (selector, label);
    }
}
