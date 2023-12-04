/*using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using System.Diagnostics;

namespace BlobGame.Drawing;
/// <summary>
/// Class for settup up and controlling the drawing of the game.
/// </summary>
internal static class RenderManager {

    /// <summary>
    /// Stopwatch to keep track of the time between frames (delta time).
    /// </summary>
    private static Stopwatch UpdateStopwatch { get; }

    /// <summary>
    /// The default font for buttons and ingame text
    /// </summary>
    internal static FontResource MainFont { get; private set; }
    /// <summary>
    /// The default font to use for the ui.
    /// </summary>
    internal static FontResource GuiFont { get; private set; }

    /// <summary>
    /// Keeps trakc of the time since the game started.
    /// </summary>
    internal static float Time { get; private set; }

    /// <summary>
    /// Static constructor to initialize clear color and required properties.
    /// </summary>
    static RenderManager() {
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
        MainFont = ResourceManager.FontLoader.Get("main");
        GuiFont = ResourceManager.FontLoader.Get("gui");
    }

    /// <summary>
    /// Main drawing method. Called every frame. Tracks delta time and calls the game's draw method. Also scales all drawing operations to the game's resolution.
    /// </summary>
    internal static void Draw() {


    }
}
*/