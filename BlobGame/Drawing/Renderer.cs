using BlobGame.ResourceHandling;
using BlobGame.Util;
using Raylib_CsLo;
using System.Diagnostics;
using System.Numerics;

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
    /// Tubmler to draw cute little strawberries in the background.
    /// </summary>
    private static StrawberryBackgroundTumbler Tumbler { get; }

    /// <summary>
    /// Static constructor to initialize clear color and required properties.
    /// </summary>
    static Renderer() {
        UpdateStopwatch = new Stopwatch();
        Time = 0;
        Tumbler = new StrawberryBackgroundTumbler(60);
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
        Tumbler.Load();
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
        DrawBackground();
        Tumbler.Draw(dT);


        Game.GameManager.Draw();

        RlGl.rlPopMatrix();

        if (Application.DRAW_DEBUG) {
            int fps = Raylib.GetFPS();
            Raylib.DrawText(fps.ToString(), 10, 10, 12, Raylib.LIME);
        }

        Raylib.EndDrawing();
    }

    private static void DrawBackground() {
        const float ANGLE = -12.5f;
        Color elementColor = ResourceManager.GetColor("light_accent").Resource.ChangeAlpha(64);
        //Color elementColor = new Color(255, 255, 255, 64);

        Raylib.DrawRectanglePro(
            new Rectangle(-100, 287.5f, 2500, 100),
            new Vector2(), ANGLE, elementColor);

        Raylib.DrawRectanglePro(
            new Rectangle(-100, Application.BASE_HEIGHT * 0.80f, 2500, 25),
            new Vector2(), ANGLE, elementColor);

        Raylib.DrawRectanglePro(
            new Rectangle(-100, Application.BASE_HEIGHT * 0.85f, 2500, 200),
            new Vector2(), ANGLE, elementColor);
    }
}
