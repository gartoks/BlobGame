using BlobGame.Drawing;
using BlobGame.Game.Gui;
using BlobGame.Game.Util;
using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using static BlobGame.Game.Gui.GUISelector;
using static BlobGame.Game.Util.Settings;

namespace BlobGame.Game.Scenes;
internal class SettingsScene : Scene {
    private GUIPanel BackgroundPanel { get; }

    private GUITextButton BackButton { get; }
    private GUITextButton ApplyButton { get; }

    private GUILabel ScreenModeLabel { get; }
    private GUISelector ScreenModeSelector { get; }

    private GUILabel ResolutionLabel { get; }
    private GUISelector ResolutionSelector { get; }

    private GUILabel MonitorLabel { get; }
    private GUISelector MonitorSelector { get; }

    //private (int w, int h) Resolution { get; set; }
    //private eScreenMode ScreenMode { get; set; }
    //private int Monitor { get; set; }

    internal override void Load() {
        //Resolution = Application.Settings.GetCurrentResolution();
        //ScreenMode = Application.Settings.ScreenMode;
        //Monitor = Application.Settings.GetCurrentMonitor();
    }

    public SettingsScene() {
        BackgroundPanel = new GUIPanel(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.05f,
            Application.BASE_WIDTH * 0.9f, Application.BASE_HEIGHT * 0.8f,
            Renderer.MELBA_LIGHT_PINK,
            new Vector2(0, 0));

        BackButton = new GUITextButton(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Back",
            new Vector2(0, 1));

        ApplyButton = new GUITextButton(
            Application.BASE_WIDTH * 0.95f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Apply",
            new Vector2(1, 1));

        float xOffset = 0.1f;
        MonitorLabel = new GUILabel(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f,
            "Monitor",
            new Vector2(0, 0));

        SelectionElement[] monitorElements = Application.Settings.GetMonitors()
            .Select(m => new SelectionElement($"{m.monitor}: {m.name}", m.monitor)).ToArray();

        int selectedMonitorIndex = Array.FindIndex(monitorElements, e => e.Element.Equals(Application.Settings.GetCurrentMonitor()));
        MonitorSelector = new GUISelector(
            Application.BASE_WIDTH * 0.35f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT / 16f,
            monitorElements, selectedMonitorIndex < 0 ? 0 : selectedMonitorIndex,
            new Vector2(0, 0));
        xOffset += 0.1f;

        ScreenModeLabel = new GUILabel(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f,
            "Screen Mode",
            new Vector2(0, 0));

        SelectionElement[] screenModeElements = Enum.GetValues<eScreenMode>()
            .Select(sm => new SelectionElement(sm.ToString(), sm)).ToArray();

        int selectedScreenModeIndex = Array.FindIndex(screenModeElements, e => e.Element.Equals(Application.Settings.ScreenMode));
        ScreenModeSelector = new GUISelector(
            Application.BASE_WIDTH * 0.35f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT / 16f,
            screenModeElements, selectedScreenModeIndex < 0 ? 0 : selectedScreenModeIndex,
            new Vector2(0, 0));
        xOffset += 0.1f;

        ResolutionLabel = new GUILabel(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f,
            "Resolution",
            new Vector2(0, 0));

        SelectionElement[] resolutionElements = Settings.AVAILABLE_RESOLUTIONS
            .Select(res => new SelectionElement($"{res.w}x{res.h}", res)).ToArray();

        int selectedResolutionIndex = Array.FindIndex(resolutionElements, e => e.Element.Equals(Application.Settings.GetCurrentResolution()));
        ResolutionSelector = new GUISelector(
            Application.BASE_WIDTH * 0.35f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT / 16f,
            resolutionElements, selectedResolutionIndex < 0 ? 0 : selectedResolutionIndex,
            new Vector2(0, 0));
        xOffset += 0.1f;
    }

    internal override void Draw() {
        DrawBackButton();
    }

    private void DrawBackButton() {
        BackgroundPanel.Draw();
        MonitorLabel.Draw();
        ScreenModeLabel.Draw();
        ResolutionLabel.Draw();

        if (BackButton.Draw())
            GameManager.SetScene(new MainMenuScene());
        if (ApplyButton.Draw())
            ApplySettings();

        MonitorSelector.Draw();
        ScreenModeSelector.Draw();
        ResolutionSelector.Draw();
    }

    private void ApplySettings() {
        int monitor = (int)MonitorSelector.SelectedElement.Element;
        eScreenMode screenMode = (eScreenMode)ScreenModeSelector.SelectedElement.Element;
        (int w, int h) resolution = ((int w, int h))ResolutionSelector.SelectedElement.Element;

        if (resolution != Application.Settings.GetCurrentResolution())
            Application.Settings.SetResolution(resolution.w, resolution.h);
        if (screenMode != Application.Settings.ScreenMode)
            Application.Settings.SetScreenMode(screenMode);
        if (monitor != Application.Settings.GetCurrentMonitor())
            Application.Settings.SetMonitor(monitor);

        Application.Exit();
        Process.Start(Assembly.GetExecutingAssembly().Location);
    }
}
