using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Diagnostics;
using System.Numerics;

namespace BlobGame.Drawing;
/// <summary>
/// Class for settup up and controlling the drawing of the game.
/// </summary>
internal static class Renderer {
    public static Color LIGHT_PINK { get; } = new Color(255, 209, 238, 255);
    public static Color DARK_PINK { get; } = new Color(255, 168, 223, 255);

    /// <summary>
    /// The color to clear the screen with.
    /// </summary>
    public static Color ClearColor { get; set; }

    /// <summary>
    /// Stopwatch to keep track of the time between frames (delta time).
    /// </summary>
    private static Stopwatch UpdateStopwatch { get; }

    /// <summary>
    /// The default font to use for drawing text.
    /// </summary>
    internal static FontResource Font { get; private set; }

    /// <summary>
    /// Static constructor to initialize clear color and required properties.
    /// </summary>
    static Renderer() {
        ClearColor = new Color(234, 122, 147, 255);

        UpdateStopwatch = new Stopwatch();
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
        Font = ResourceManager.DefaultFont;
    }

    static float Time = 0;
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
        Raylib.ClearBackground(ClearColor);

        RlGl.rlPushMatrix();

        //Raylib.BeginMode2D(cam);
        RlGl.rlScalef(Application.WorldToScreenMultiplierX, Application.WorldToScreenMultiplierY, 1);
        DrawBackground();
        Game.GameManager.Draw();
        //Raylib.EndMode2D();

        RlGl.rlPopMatrix();

        //GUIHandler.Draw();

        if (Application.DRAW_DEBUG) {
            int fps = Raylib.GetFPS();
            Raylib.DrawText(fps.ToString(), 10, 10, 12, Raylib.LIME);
        }

        Raylib.EndDrawing();
    }

    private static void DrawBackground() {
        Raylib.DrawRectanglePro(
            new Rectangle(-100, 287.5f, 2500, 100),
            new Vector2(), -12.5f, new Color(255, 255, 255, 64));

        Raylib.DrawRectanglePro(
            new Rectangle(-100, Application.BASE_HEIGHT * 0.80f, 2500, 25),
            new Vector2(), -12.5f, new Color(255, 255, 255, 64));

        Raylib.DrawRectanglePro(
            new Rectangle(-100, Application.BASE_HEIGHT * 0.85f, 2500, 200),
            new Vector2(), -12.5f, new Color(255, 255, 255, 64));
    }
}
