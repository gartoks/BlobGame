using BlobGame.Audio;
using BlobGame.ResourceHandling;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using SimpleGL;
using SimpleGL.Graphics;
using System.Dynamic;
using System.Text.Json;

namespace BlobGame.App;
/// <summary>
/// Class for managing the game settings
/// </summary>
internal sealed class Settings {
    /// <summary>
    /// Enum for the different screen modes.
    /// </summary>
    public enum eScreenMode { Fullscreen, Borderless, Windowed }

    /// <summary>
    /// List of supported resolutions.
    /// </summary>
    public static IReadOnlyList<(int w, int h)> AVAILABLE_RESOLUTIONS { get; } = new[] {
        (1280, 720),
        (1366, 768),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160)
    };

    /// <summary>
    /// The current screen mode.
    /// </summary>
    public eScreenMode ScreenMode { get; private set; }

    /// <summary>
    /// The current inner window resolution.
    /// </summary>
    public (int w, int h) Resolution => (Application.Window.ClientSize.X, Application.Window.ClientSize.X);

    ///// <summary>
    ///// The name of the monitor the window is on.
    ///// </summary>
    //public string Monitor => Monitors.GetMonitorFromWindow(Application.Window).Name;

    /// <summary>
    /// The music volume from 0 to 100. It is internally stored as 0 to 100 because easier comparison when adjusting settings.
    /// </summary>
    private int _MusicVolume { get; set; }
    public int MusicVolume {
        get => _MusicVolume;
        set {
            _MusicVolume = Math.Clamp(value, 0, 100);
            Save();

            AudioManager.SetMusicVolume(_MusicVolume);
        }
    }

    /// <summary>
    /// The sound volume from 0 to 100. It is internally stored as 0 to 100 because easier comparison when adjusting settings.
    /// </summary>
    private int _SoundVolume { get; set; }
    public int SoundVolume {
        get => _SoundVolume;
        set {
            _SoundVolume = Math.Clamp(value, 0, 100);
            Save();

            AudioManager.SetSoundVolume(_SoundVolume);
        }
    }

    /// <summary>
    /// Sets the state of the tutorial. If true, the tutorial will be shown when starting a new game.
    /// </summary>
    private bool _IsTutorialEnabled { get; set; }
    public bool IsTutorialEnabled {
        get => _IsTutorialEnabled;
        set {
            _IsTutorialEnabled = value;
            Save();
        }
    }

    /// The current theme.
    /// </summary>
    public string ThemeName { get; private set; }

    /// <summary>
    /// Default Constructor.
    /// </summary>
    public Settings() {
        ThemeName = "MelbaToast";
    }

    /// <summary>
    /// Sets and loads a new theme.
    /// </summary>
    /// <param name="name"></param>
    public void SetTheme(string name) {
        AudioManager.ClearSounds();
        AudioManager.ClearMusic();
        ResourceManager.SetTheme(name);
        ThemeName = name;
        Save();
    }

    /// <summary>
    /// Set the windows screen mode.
    /// </summary>
    /// <param name="mode"></param>
    public void SetScreenMode(eScreenMode mode) {
        switch (mode) {
            case eScreenMode.Fullscreen:
                Application.Window.WindowState = WindowState.Fullscreen;
                /*Application.Window.WindowBorder = WindowBorder.Hidden;
                Raylib.SetWindowSize(GetCurrentResolution().w, GetCurrentResolution().h);
                Raylib.SetWindowPosition(0, 0);*/
                break;
            case eScreenMode.Borderless:
                Application.Window.WindowBorder = WindowBorder.Hidden;
                Application.Window.ClientLocation = new(0, 0);
                Application.Window.ClientSize = Monitors.GetMonitorFromWindow(Application.Window).ClientArea.Size;
                break;
            case eScreenMode.Windowed:
                Application.Window.WindowState = WindowState.Normal;
                Application.Window.WindowBorder = WindowBorder.Fixed;
                break;
        }

        ScreenMode = mode;
        Save();
    }

    /// <summary>
    /// Sets the resolution to the given width and height. Only works if the screen mode is not borderless.
    /// </summary>
    /// <param name="w"></param>
    /// <param name="h"></param>
    public void SetResolution(int w, int h) {
        if (ScreenMode == eScreenMode.Borderless)
            return;

        Application.Window.ClientSize = new(w, h);
        Save();
    }

    /// <summary>
    /// Changes the monitor the game is displayed on. Currently disabled because funky behaviour.
    /// </summary>
    /// <param name="monitor"></param>
    public void SetMonitor(int monitor) {
        // TODO
        /*Application.Window.CurrentMonitor =

        if (monitor < 0 || monitor >= Raylib.GetMonitorCount())
            return;

        switch (ScreenMode) {
            case eScreenMode.Fullscreen:
                Raylib.SetWindowPosition(0, 0);
                break;
            case eScreenMode.Borderless:
                Raylib.SetWindowSize(GetMonitorResolution(monitor).w, GetMonitorResolution(monitor).h);
                Raylib.SetWindowPosition(0, 0);
                break;
            case eScreenMode.Windowed:
                int relativeWindowsPosX = (int)(GetMonitorResolution(monitor).w * Raylib.GetWindowPosition().X / Raylib.GetScreenWidth());
                int relativeWindowsPosY = (int)(GetMonitorResolution(monitor).h * Raylib.GetWindowPosition().Y / Raylib.GetScreenHeight());
                Raylib.SetWindowPosition(relativeWindowsPosX, relativeWindowsPosY);
                break;
            default:
                break;
        }

        Raylib.SetWindowMonitor(monitor);
        Save();*/
    }

    /// <summary>
    /// Gets the indices and names of all monitors.
    /// </summary>
    /// <returns></returns>
    public (int index, string name)[] GetMonitors() {
        return GraphicsHelper.GetMonitors().Select((m, i) => (i, m.Name)).ToArray();
    }

    /// <summary>
    /// Gets the resolution of the monitor the game is currently displayed on.
    /// </summary>
    public (int w, int h) GetCurrentMonitorResolution() {
        MonitorInfo monitor = Application.Window.GetMonitor();
        return (monitor.HorizontalResolution, monitor.VerticalResolution);
    }

    /// <summary>
    /// Saves the settings to a file.
    /// </summary>
    private void Save() {
        string file = Files.GetConfigFilePath("settings.json");

        dynamic settingsData = new ExpandoObject();

        settingsData.ScreenMode = ScreenMode.ToString();
        settingsData.Monitor = GraphicsHelper.GetMonitors().ToList().IndexOf(Application.Window.GetMonitor());
        settingsData.ResolutionW = Application.Window.ClientSize.X;
        settingsData.ResolutionH = Application.Window.ClientSize.Y;
        settingsData.MusicVolume = _MusicVolume;
        settingsData.SoundVolume = _SoundVolume;
        settingsData.IsTutorialEnabled = _IsTutorialEnabled;
        settingsData.ThemeName = ThemeName;

        File.WriteAllText(file, JsonSerializer.Serialize(settingsData));
    }

    /// <summary>
    /// Loads the settings from a file. If the file does not exist or is invalid, the default settings are used.
    /// </summary>
    public void Load() {
        //int monitor = GraphicsHelper.GetMonitors().ToList().IndexOf(Application.Window.GetMonitor());    // TODO TEST
        (int w, int h) resolution = AVAILABLE_RESOLUTIONS[0];
        eScreenMode screenMode = eScreenMode.Windowed;
        int musicVolume = 100;
        int soundVolume = 100;
        bool isTutorialEnabled = true;
        string themeName = "MelbaToast";

        string file = Files.GetConfigFilePath("settings.json");
        if (File.Exists(file)) {
            SettingsData? settingsData = JsonSerializer.Deserialize<SettingsData>(File.ReadAllText(file));

            if (settingsData != null) {
                if (Enum.TryParse(settingsData!.ScreenMode, out eScreenMode screenMode2))
                    screenMode = screenMode2;

                //monitor = settingsData.Monitor;
                resolution = (settingsData.ResolutionW, settingsData.ResolutionH);
                musicVolume = settingsData.MusicVolume;
                soundVolume = settingsData.SoundVolume;
                isTutorialEnabled = settingsData.IsTutorialEnabled;
                themeName = settingsData.ThemeName;
            }
        }

        SetScreenMode(screenMode);
        SetResolution(resolution.w, resolution.h);
        //SetMonitor(monitor);
        MusicVolume = musicVolume;
        SoundVolume = soundVolume;
        IsTutorialEnabled = isTutorialEnabled;
        SetTheme(themeName);
        Save();
    }

    private record SettingsData(
        string ScreenMode, int Monitor, int ResolutionW, int ResolutionH,
        int MusicVolume = 100, int SoundVolume = 100,
        bool IsTutorialEnabled = true,
        string ThemeName = "default"
        );
}
