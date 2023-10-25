using BlobGame.Game;
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Diagnostics;

namespace BlobGame.Drawing;
internal static class DrawHandler {
    public static Color ClearColor { get; set; }
    public static bool DrawFPS { get; set; }

    private static Stopwatch UpdateStopwatch { get; }

    internal static FontResource Font { get; private set; }

    static DrawHandler() {
        ClearColor = new Color(234, 122, 147, 255);
        //ClearColor = new Color(32, 32, 32, 255);
        DrawFPS = true;

        UpdateStopwatch = new Stopwatch();
    }

    internal static void Initialize() {
    }

    internal static void Load() {
        Font = ResourceHandler.DefaultFont;
    }

    static float Time = 0;
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
        RlGl.rlScalef(RaylibApp.WorldToScreenMultiplierX, RaylibApp.WorldToScreenMultiplierY, 1);
        GameHandler.Draw();
        //Raylib.EndMode2D();

        RlGl.rlPopMatrix();

        //GUIHandler.Draw();

        if (DrawFPS) {
            int fps = Raylib.GetFPS();
            Raylib.DrawText(fps.ToString(), 10, 10, 12, Raylib.LIME);
        }

        Raylib.EndDrawing();
    }
}
