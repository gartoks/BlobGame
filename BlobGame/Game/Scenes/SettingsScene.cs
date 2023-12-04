using BlobGame.App;
using BlobGame.Game.Gui;
using OpenTK.Mathematics;
using System.Diagnostics;
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
        BackgroundPanel = new GuiNPatchPanel("0.05 0.05 0.9 0.8", "panel", 1, new Vector2(0, 0));

        BackButton = new GuiTextButton(
            GameApplication.PROJECTION_WIDTH * 0.05f, GameApplication.PROJECTION_HEIGHT * 0.95f,
            GameApplication.PROJECTION_WIDTH / 8f, GameApplication.PROJECTION_HEIGHT / 16f,
            "Back",
            2,
            new Vector2(0, 1));

        ApplyButton = new GuiTextButton(
            GameApplication.PROJECTION_WIDTH * 0.95f, GameApplication.PROJECTION_HEIGHT * 0.95f,
            GameApplication.PROJECTION_WIDTH / 8f, GameApplication.PROJECTION_HEIGHT / 16f,
            "Apply",
            2,
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
            Array.FindIndex(Enum.GetValues<eScreenMode>(), sm => sm == GameApplication.Settings.ScreenMode));
        ScreenModeLabel = screenModeLabel;
        ScreenModeSelector = screenModeSelector;
        xOffset += 0.1f;

        (GuiSelector resolutionSelector, GuiLabel resolutionLabel) = CreateSettingsEntry(
            "Resolution", xOffset,
            Settings.AVAILABLE_RESOLUTIONS.Select(res => new SelectionElement($"{res.w}x{res.h}", res)).ToArray(),
            Array.FindIndex<(int, int)>(Settings.AVAILABLE_RESOLUTIONS.ToArray(), r => r.Equals(GameApplication.Settings.Resolution)));
        ResolutionLabel = resolutionLabel;
        ResolutionSelector = resolutionSelector;
        xOffset += 0.1f;

        (GuiSelector musicVolumeSelector, GuiLabel musicVolumeLabel) = CreateSettingsEntry(
            "Music Volume", xOffset,
            Enumerable.Range(0, 11).Select(i => new SelectionElement($"{i * 10f}%", i * 10)).ToArray(),
            GameApplication.Settings.MusicVolume / 10);
        MusicVolumeLabel = musicVolumeLabel;
        MusicVolumeSelector = musicVolumeSelector;
        xOffset += 0.1f;

        (GuiSelector soundVolumeSelector, GuiLabel soundVolumeLabel) = CreateSettingsEntry(
            "Sound Volume", xOffset,
            Enumerable.Range(0, 11).Select(i => new SelectionElement($"{i * 10f}%", i * 10)).ToArray(),
            GameApplication.Settings.SoundVolume / 10);
        SoundVolumeLabel = soundVolumeLabel;
        SoundVolumeSelector = soundVolumeSelector;
        xOffset += 0.1f;

        SelectionElement[] availableThemes = Directory.GetFiles(Files.GetResourceFilePath())
            .Where(file => file.EndsWith(".theme"))
            .Select(file => new SelectionElement($"{Path.GetFileNameWithoutExtension(file)}", Path.GetFileNameWithoutExtension(file))).ToArray();
        int selectedThemeIndex = Array.FindIndex(availableThemes, e => e.Element.Equals(GameApplication.Settings.ThemeName));

        (GuiSelector themeSelector, GuiLabel themeLabel) = CreateSettingsEntry(
            "Theme", xOffset,
            availableThemes,
            selectedThemeIndex);
        ThemeSelector = themeSelector;
        ThemeLabel = themeLabel;
        xOffset += 0.1f;

        ResetScoreButton = new GuiTextButton("0.1 0.8 0.25 0.0625", "Reset Score", 3, new Vector2(0, 1));
        ResetTutorialButton = new GuiTextButton("0.85 0.8 0.25 0.0625", "Reset Tutorial", 3, new Vector2(1, 1));
    }

    internal override void Load() {
        LoadAllGuiElements();
    }

    internal override void Update(float dT) {
        base.Update(dT);

        if (BackButton.IsClicked)
            GameManager.SetScene(new MainMenuScene());
        if (ApplyButton.IsClicked)
            ApplySettings();

        if (ResetScoreButton.IsClicked)
            GameManager.Scoreboard.Reset();
        if (ResetTutorialButton.IsClicked)
            GameApplication.Settings.IsTutorialEnabled = true;
    }

    internal override void Render() {
    }

    internal override void RenderGui() {
        base.RenderGui();

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
    }

    private void ApplySettings() {
        //int monitor = (int)MonitorSelector.SelectedElement.Element;
        eScreenMode screenMode = (eScreenMode)ScreenModeSelector.SelectedElement.Element;
        (int w, int h) resolution = ((int w, int h))ResolutionSelector.SelectedElement.Element;
        int soundVolume = (int)SoundVolumeSelector.SelectedElement.Element;
        int musicVolume = (int)MusicVolumeSelector.SelectedElement.Element;
        string theme = (string)ThemeSelector.SelectedElement.Element;

        bool needsRestart = false;
        if (resolution != GameApplication.Settings.Resolution) {
            GameApplication.Settings.SetResolution(resolution.w, resolution.h);
            needsRestart = true;
        }
        if (screenMode != GameApplication.Settings.ScreenMode) {
            GameApplication.Settings.SetScreenMode(screenMode);
            needsRestart = true;
        }
        //if (monitor != Application.Settings.GetCurrentMonitor()) {
        //    Application.Settings.SetMonitor(monitor);
        //    needsRestart = true;
        //}
        if (soundVolume != GameApplication.Settings.SoundVolume)
            GameApplication.Settings.SoundVolume = soundVolume;
        if (musicVolume != GameApplication.Settings.MusicVolume)
            GameApplication.Settings.MusicVolume = musicVolume;
        if (theme != GameApplication.Settings.ThemeName)
            GameApplication.Settings.SetTheme(theme);

#if WINDOWS

        if (needsRestart) {
            // This is needed because if the resolution, screen mode or monitor is change the UI is all fricked up
            GameApplication.Exit();
            Process.Start(Assembly.GetExecutingAssembly().Location);
        }

#endif
        //GameManager.SetScene(new SettingsScene());
    }

    private (GuiSelector, GuiLabel) CreateSettingsEntry(string title, float xOffset, SelectionElement[] selectionElements, int selectedIndex) {
        GuiLabel label = new GuiLabel($"0.1 {xOffset} 0.25 {1f / 10f}", title, 3, new Vector2(0, 0.5f));
        label.TextAlignment = eTextAlignment.Center;
        //label.DrawOutline = true; //TODO

        GuiSelector selector = new GuiSelector($"0.35 {xOffset} 0.5 {1f / 16f}", selectionElements, selectedIndex < 0 ? 0 : selectedIndex, 3, new Vector2(0, 0.5f));

        return (selector, label);
    }
}
