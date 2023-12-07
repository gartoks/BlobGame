using BlobGame.App;
using BlobGame.Audio;
using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Diagnostics;
using System.Reflection;

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
    private const string NAME = "Toasted";
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
    /// Property to access the game's settings.
    /// </summary>
    public static Settings Settings { get; }

    /// <summary>
    /// Static constructor. Initializes the game state, creates threads and hides the console window.
    /// </summary>
    static Application() {
        IsRunning = false;
        GameThread = new Thread(RunGameThread);

        Settings = new Settings();
    }

    /// <summary>
    /// Calls the initialization methods of all components.
    /// </summary>
    public static void Initialize() {
        ResourceManager.Initialize();
        AudioManager.Initialize();
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

        Raylib.SetConfigFlags(ConfigFlags.FLAG_MSAA_4X_HINT);
        Raylib.InitWindow(BASE_WIDTH, BASE_HEIGHT, NAME);
        Raylib.SetTargetFPS(FPS);
        Raylib.SetExitKey(KeyboardKey.KEY_NULL);
        Raylib.InitAudioDevice();

        Image icon = LoadIcon();
        Raylib.SetWindowIcon(icon);

        Settings.Load();
        ResourceManager.Load();
        AudioManager.Load();
        Input.Load();
        Renderer.Load();
        Game.GameManager.Load();

        GameThread.Start();

        while (IsRunning) {
            ResourceManager.Update();
            AudioManager.Update();
            Input.Update();
            Renderer.Draw();

            if (Raylib.WindowShouldClose()) {
                IsRunning = false;
            }
        }

        ResourceManager.Unload();

        GameThread.Join();
        Raylib.CloseAudioDevice();
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

    private static Image LoadIcon() {
        Assembly assembly = Assembly.GetExecutingAssembly();
        string resourceName = "BlobGame.Resources.icon.png";

        using Stream stream = assembly.GetManifestResourceStream(resourceName)!;
        byte[] imageData;
        using (MemoryStream ms = new MemoryStream()) {
            stream.CopyTo(ms);
            ms.Position = 0;
            imageData = ms.ToArray();
        }

        Image image;
        unsafe {
            fixed (byte* imagePtr = imageData) {
                image = Raylib.LoadImageFromMemory(".png", imagePtr, imageData.Length);
            }
        }

        return image;
    }

}
