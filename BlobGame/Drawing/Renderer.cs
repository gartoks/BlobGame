using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Diagnostics;

namespace BlobGame.Drawing;
/// <summary>
/// Class for settup up and controlling the drawing of the game.
/// </summary>
internal static class Renderer {

    /// <summary>
    /// Stopwatch to keep track of the time between frames (delta time).
    /// </summary>
    private static Stopwatch UpdateStopwatch { get; }

    /// <summary>
    /// The default font to use for drawing text.
    /// </summary>
    internal static FontResource Font { get; private set; }

    /// <summary>
    /// Keeps trakc of the time since the game started.
    /// </summary>
    internal static float Time { get; private set; }

    /// <summary>
    /// Static constructor to initialize clear color and required properties.
    /// </summary>
    static Renderer() {
        UpdateStopwatch = new Stopwatch();
        Time = 0;
    }

    /// <summary>
    /// Initializes the drawing. Currently does nothing.
    /// </summary>
    internal static void Initialize() {
    }

    /// <summary>
    /// Loads global resources.
    /// </summary>
    internal static void Load() {
        Font = ResourceManager.GetFont("NewBread");
    }

    /// <summary>
    /// Main drawing method. Called every frame. Tracks delta time and calls the game's draw method. Also scales all drawing operations to the game's resolution.
    /// </summary>
    internal static void Draw() {
        UpdateStopwatch.Stop();
        long ms = UpdateStopwatch.ElapsedMilliseconds;
        float dT = ms / 1000f;
        Time += dT;
        UpdateStopwatch.Restart();

        Raylib.BeginDrawing();
        Raylib.ClearBackground(ResourceManager.GetColor("background").Resource);

        RlGl.rlPushMatrix();

        RlGl.rlScalef(Application.WorldToScreenMultiplierX, Application.WorldToScreenMultiplierY, 1);

        Game.GameManager.Draw(dT);

        RlGl.rlPopMatrix();

        if (Application.DRAW_DEBUG) {
            int fps = Raylib.GetFPS();
            Raylib.DrawText(fps.ToString(), 10, 10, 16, Raylib.LIME);

            float x = Raylib.GetMouseX() / (float)Raylib.GetRenderWidth();
            float y = Raylib.GetMouseY() / (float)Raylib.GetRenderHeight();
            Raylib.DrawText($"{Raylib.GetMousePosition()}, ({x:0.000}, {y:0.000})", 30, 30, 16, Raylib.MAGENTA);
        }

        Raylib.EndDrawing();
    }
}
