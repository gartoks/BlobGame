using BlobGame.Util;
using SimpleGL;
using SimpleGL.Graphics;
using System.Runtime.InteropServices;

namespace BlobGame.App;
/// <summary>
/// The main application class. Initializes the window, drawing engine and controllers. Manages the game loop and threads.
/// </summary>
internal sealed partial class GameApplication : Application {
    /// <summary>
    /// Set to true to enable debug drawing.
    /// </summary>
    internal static bool DRAW_DEBUG = false;

    /// <summary>
    /// The name of the application. Used for the window title.
    /// </summary>
    private const string NAME = "Blob Game";
    /// <summary>
    /// The frames per second the game is targeting.
    /// </summary>
    private const int FPS = 60;
    /// <summary>
    /// The udpates per second the game is targeting.
    /// </summary>
    private const int UPS = 60;
    /// <summary>
    /// The resource loading thread checks per second.
    /// </summary>
    private const int RPS = 10;

    /// <summary>
    /// The base width of the game window. All resolutions use this to scale components to the right proportions.
    /// </summary>
    internal const int PROJECTION_WIDTH = 1920;
    /// <summary>
    /// The base height of the game window. All resolutions use this to scale components to the right proportions.
    /// </summary>
    internal const int PROJECTION_HEIGHT = 1080;

    /// <summary>
    /// Property to access the game's settings.
    /// </summary>
    public static Settings Settings { get; }

    /// <summary>
    /// The main camera of the game.
    /// </summary>
    public static Camera Camera { get; }

    /// <summary>
    /// Static constructor. Initializes the game state, creates threads and hides the console window.
    /// </summary>
    static GameApplication() {
        nint handle = GetConsoleWindow();
        ShowWindow(handle, SW_HIDE);

        Settings = new Settings();
        Camera = new Camera();
    }

    public GameApplication()
        : base(FPS, UPS) {

        GameRenderer = new Renderer();
        GuiRenderer = new Renderer();
    }

    protected override void OnInitialize() {
        base.OnInitialize();

        ThreadManager.RegisterGameThread(new ResourceThreadBase(RPS));
    }

#if WINDOWS
    [DllImport("kernel32.dll")]
    static extern IntPtr GetConsoleWindow();

    [DllImport("user32.dll")]
    static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
#else
    // Linux (and macOS probably) don't open a console window by default
    static IntPtr GetConsoleWindow() {
        return 0;
    }
    static bool ShowWindow(IntPtr hWnd, int nCmdShow) {
        return false;
    }
#endif

    const int SW_HIDE = 0;
    const int SW_SHOW = 5;
}
