using BlobGame.App;
using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using static BlobGame.App.Settings;
using static BlobGame.Game.Gui.GuiSelector;

namespace BlobGame.Game.Scenes;
internal class SettingsScene : Scene {
    private GuiNPatchPanel BackgroundPanel { get; }

    private GuiLabel ScreenModeLabel { get; }
    private GuiSelector ScreenModeSelector { get; }

    private GuiLabel ResolutionLabel { get; }
    private GuiSelector ResolutionSelector { get; }

    private GuiLabel MusicVolumeLabel { get; }
    private GuiSelector MusicVolumeSelector { get; }

    private GuiLabel SoundVolumeLabel { get; }
    private GuiSelector SoundVolumeSelector { get; }

    //private GuiLabel MonitorLabel { get; }
    //private GuiSelector MonitorSelector { get; }

    private GuiLabel ThemeLabel { get; }
    private GuiSelector ThemeSelector { get; }

    private GuiTextButton ResetScoreButton { get; }
    private GuiTextButton ResetTutorialButton { get; }

    private GuiTextButton ApplyButton { get; }
    private GuiTextButton BackButton { get; }


    public SettingsScene() {
        BackgroundPanel = new GuiNPatchPanel("0.05 0.05 0.9 0.8", "panel", new Vector2(0, 0));

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

        float xOffset = 0.175f;
        /*(GuiSelector monitorSelector, GuiLabel monitorLabel) = CreateSettingsEntry(
            "Monitor", xOffset,
            Application.Settings.GetMonitors().Select(m => new SelectionElement($"{m.monitor}: {m.name}", m.monitor)).ToArray(),
            Array.FindIndex(Application.Settings.GetMonitors(), m => m.monitor == Application.Settings.GetCurrentMonitor()));
        MonitorLabel = monitorLabel;
        MonitorSelector = monitorSelector; */
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

        SelectionElement[] availableThemes = Directory.GetFiles(Files.GetResourceFilePath())
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

        ResetScoreButton = new GuiTextButton("0.1 0.8 0.25 0.0625", "Reset Score", new Vector2(0, 1));
        ResetTutorialButton = new GuiTextButton("0.85 0.8 0.25 0.0625", "Reset Tutorial", new Vector2(1, 1));
    }

    internal override void Load() {
        LoadAllGuiElements();
    }

    internal override void Draw() {
        BackgroundPanel.Draw();
        BackButton.Draw();
        ApplyButton.Draw();

        //MonitorLabel.Draw(); // TODO: Currently disabled because it is not working.
        ScreenModeLabel.Draw();
        ResolutionLabel.Draw();
        MusicVolumeLabel.Draw();
        SoundVolumeLabel.Draw();
        ThemeLabel.Draw();

        //MonitorSelector.Draw();   // TODO: Currently disabled because it is not working.
        ScreenModeSelector.Draw();
        ResolutionSelector.Draw();
        MusicVolumeSelector.Draw();
        SoundVolumeSelector.Draw();
        ThemeSelector.Draw();

        ResetScoreButton.Draw();
        ResetTutorialButton.Draw();

        if (BackButton.IsClicked)
            GameManager.SetScene(new MainMenuScene());
        if (ApplyButton.IsClicked)
            ApplySettings();

        if (ResetScoreButton.IsClicked)
            GameManager.Scoreboard.Reset();
        if (ResetTutorialButton.IsClicked)
            Application.Settings.IsTutorialEnabled = true;
    }

    private void ApplySettings() {
        //int monitor = (int)MonitorSelector.SelectedElement.Element;
        eScreenMode screenMode = (eScreenMode)ScreenModeSelector.SelectedElement.Element;
        (int w, int h) resolution = ((int w, int h))ResolutionSelector.SelectedElement.Element;
        int soundVolume = (int)SoundVolumeSelector.SelectedElement.Element;
        int musicVolume = (int)MusicVolumeSelector.SelectedElement.Element;
        string theme = (string)ThemeSelector.SelectedElement.Element;

        bool needsRestart = false;
        if (resolution != Application.Settings.GetCurrentResolution()) {
            Application.Settings.SetResolution(resolution.w, resolution.h);
            needsRestart = true;
        }
        if (screenMode != Application.Settings.ScreenMode) {
            Application.Settings.SetScreenMode(screenMode);
            needsRestart = true;
        }
        //if (monitor != Application.Settings.GetCurrentMonitor()) {
        //    Application.Settings.SetMonitor(monitor);
        //    needsRestart = true;
        //}
        if (soundVolume != Application.Settings.SoundVolume)
            Application.Settings.SoundVolume = soundVolume;
        if (musicVolume != Application.Settings.MusicVolume)
            Application.Settings.MusicVolume = musicVolume;
        if (theme != Application.Settings.GetCurrentThemeName())
            Application.Settings.SetTheme(theme);

#if WINDOWS

        if (needsRestart) {
            // This is needed because if the resolution, screen mode or monitor is change the UI is all fricked up
            Application.Exit();
            Process.Start(Assembly.GetExecutingAssembly().Location);
        }

#endif
        //GameManager.SetScene(new SettingsScene());
    }

    private (GuiSelector, GuiLabel) CreateSettingsEntry(string title, float xOffset, SelectionElement[] selectionElements, int selectedIndex) {
        GuiLabel label = new GuiLabel($"0.135 {xOffset} 0.25 {1f / 10f}", title, new Vector2(0, 0.5f));
        label.TextAlignment = eTextAlignment.Left;
        label.Color = ResourceManager.ColorLoader.Get("font_dark");

        GuiSelector selector = new GuiSelector($"0.35 {xOffset} 0.5 {1f / 16f}",
            selectionElements, selectedIndex < 0 ? 0 : selectedIndex,
            new Vector2(0, 0.5f));

        return (selector, label);
    }
}
