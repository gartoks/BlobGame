using BlobGame.App;
using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace BlobGame;
/// <summary>
/// The main application class. Initializes the window, drawing engine and controllers. Manages the game loop and threads.
/// </summary>
internal static class Application {
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
    /// The base width of the game window. All resolutions use this to scale components to the right proportions.
    /// </summary>
    internal const int BASE_WIDTH = 1920;
    /// <summary>
    /// The base height of the game window. All resolutions use this to scale components to the right proportions.
    /// </summary>
    internal const int BASE_HEIGHT = 1080;

    /// <summary>
    /// Multiplier to convert world coordinates to screen coordinates. Used to scale components to the right proportions.
    /// </summary>
    public static float WorldToScreenMultiplierX => Raylib.GetScreenWidth() / (float)BASE_WIDTH;
    /// <summary>
    /// Multiplier to convert world coordinates to screen coordinates. Used to scale components to the right proportions.
    /// </summary>
    public static float WorldToScreenMultiplierY => Raylib.GetScreenHeight() / (float)BASE_HEIGHT;

    /// <summary>
    /// The thread the game logic is running on.
    /// </summary>
    private static Thread GameThread { get; }

    /// <summary>
    /// Flag indicating whether the game is running.
    /// </summary>
    public static bool IsRunning { get; private set; }

    /// <summary>
    /// Static constructor. Initializes the game state, creates threads and hides the console window.
    /// </summary>
    static Application() {
        IsRunning = false;
        GameThread = new Thread(RunGameThread);

        nint handle = GetConsoleWindow();
        ShowWindow(handle, SW_HIDE);
    }

    /// <summary>
    /// Calls the initialization methods of all components.
    /// </summary>
    public static void Initialize() {
        ResourceManager.Initialize();
        Input.Initialize();
        Renderer.Initialize();
        //GUIHandler.Initialize();
        Game.GameManager.Initialize();
    }

    /// <summary>
    /// Starts the game loop and threads. Also creates the window. Handles the shutdown of all components.
    /// </summary>
    public static void Start() {
        if (IsRunning)
            return;

        IsRunning = true;

        Raylib.InitWindow(BASE_WIDTH, BASE_HEIGHT, NAME);
        Raylib.SetTargetFPS(FPS);
        Raylib.SetExitKey(KeyboardKey.KEY_NULL);

        ResourceManager.Load();
        Input.Load();
        Renderer.Load();
        //GUIHandler.Load();
        Game.GameManager.Load();

        GameThread.Start();

        while (IsRunning) {
            ResourceManager.Update();
            Input.Update();
            Renderer.Draw();
        }

        GameThread.Join();
        Raylib.CloseWindow();
    }

    internal static void Exit() {
        IsRunning = false;
    }

    /// <summary>
    /// Starts and runs the game's logic loop. Including calculating delta time and ensuring thread sleep.
    /// </summary>
    private static void RunGameThread() {
        const float BASE_DELTA_TIME = 1f / UPS;
        float deltaTime = BASE_DELTA_TIME;
        Stopwatch sw = new Stopwatch();
        while (IsRunning) {
            sw.Restart();

            Game.GameManager.Update(deltaTime);

            sw.Stop();
            int sleepTime = (int)Math.Max(0, 1000 / UPS - sw.ElapsedMilliseconds);
            deltaTime = MathF.Max(BASE_DELTA_TIME, sw.ElapsedMilliseconds / 1000f);
            Thread.Sleep(sleepTime);
        }

        Game.GameManager.Unload();
    }

#if WINDOWS
    [DllImport("kernel32.dll")]
    static extern IntPtr GetConsoleWindow();

    [DllImport("user32.dll")]
    static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
#else
    // Linux (and macOS probably) don't open a console window by default
    static IntPtr GetConsoleWindow(){
        return 0;
    }
    static bool ShowWindow(IntPtr hWnd, int nCmdShow){
        return false;
    }
#endif


    const int SW_HIDE = 0;
    const int SW_SHOW = 5;
}
