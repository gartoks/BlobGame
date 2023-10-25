using BlobGame.App;
using BlobGame.Drawing;
using BlobGame.Game;
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace BlobGame;
internal static class RaylibApp {
    internal static bool DRAW_DEBUG = false;

    private const string NAME = "Blob Game";
    private const int FPS = 60;
    private const int UPS = 60;

    internal const int BASE_WIDTH = 1920;
    internal const int BASE_HEIGHT = 1080;

    public static float WorldToScreenMultiplierX => Raylib.GetScreenWidth() / (float)BASE_WIDTH;
    public static float WorldToScreenMultiplierY => Raylib.GetScreenHeight() / (float)BASE_HEIGHT;

    private static Thread GameThread { get; }

    public static bool IsRunning { get; private set; }

    static RaylibApp() {
        IsRunning = false;
        GameThread = new Thread(RunGameThread);

        nint handle = GetConsoleWindow();
        ShowWindow(handle, SW_HIDE);
    }

    public static void Initialize() {
        ResourceHandler.Initialize();
        InputHandler.Initialize();
        DrawHandler.Initialize();
        //GUIHandler.Initialize();
        GameHandler.Initialize();
    }

    public static void Start() {
        if (IsRunning)
            return;

        IsRunning = true;

        Raylib.InitWindow(BASE_WIDTH, BASE_HEIGHT, NAME);
        Raylib.SetTargetFPS(FPS);

        ResourceHandler.Load();
        InputHandler.Load();
        DrawHandler.Load();
        //GUIHandler.Load();
        GameHandler.Load();

        GameThread.Start();

        while (!Raylib.WindowShouldClose()) {
            ResourceHandler.Update();
            InputHandler.Update();
            DrawHandler.Draw();
        }

        IsRunning = false;

        GameThread.Join();

        Raylib.CloseWindow();
    }

    private static void RunGameThread() {
        const float BASE_DELTA_TIME = 1f / UPS;
        float deltaTime = BASE_DELTA_TIME;
        Stopwatch sw = new Stopwatch();
        while (IsRunning) {
            sw.Restart();

            GameHandler.Update(deltaTime);

            sw.Stop();
            int sleepTime = (int)Math.Max(0, 1000 / UPS - sw.ElapsedMilliseconds);
            deltaTime = MathF.Max(BASE_DELTA_TIME, sw.ElapsedMilliseconds / 1000f);
            Thread.Sleep(sleepTime);
        }
    }

    [DllImport("kernel32.dll")]
    static extern IntPtr GetConsoleWindow();

    [DllImport("user32.dll")]
    static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    const int SW_HIDE = 0;
    const int SW_SHOW = 5;
}
