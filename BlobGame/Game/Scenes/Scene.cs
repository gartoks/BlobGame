using BlobGame.Game.Gui;
using System.Reflection;

namespace BlobGame.Game.Scenes;
/// <summary>
/// Represents a base class for game scenes. Provides methods for scene lifecycle including loading, updating, drawing, and unloading.
/// </summary>
internal abstract class Scene {
    /// <summary>
    /// Called when the scene is loaded. Override this method to provide custom scene initialization logic and to load resources.
    /// </summary>
    internal virtual void Load() { }

    /// <summary>
    /// Called every frame to update the scene's state. 
    /// </summary>
    /// <param name="dT">The delta time since the last frame, typically used for frame-rate independent updates.</param>
    internal virtual void Update(float dT) { }

    /// <summary>
    /// Called every frame to draw the scene. Override this method to provide custom scene rendering logic.
    /// </summary>
    internal virtual void Draw() { }

    /// <summary>
    /// Called when the scene is about to be unloaded or replaced by another scene. Override this method to provide custom cleanup or deinitialization logic and to unload resources.
    /// </summary>
    internal virtual void Unload() { }

    /// <summary>
    /// Load all gui elements that are defined as properties in the scene.
    /// </summary>
    protected void LoadAllGuiElements() {
        PropertyInfo[] properties = GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        IEnumerable<PropertyInfo> guiElements = properties.Where(p => p.PropertyType.IsSubclassOf(typeof(GuiElement)));

        foreach (PropertyInfo p in guiElements) {
            GuiElement? element = p.GetValue(this) as GuiElement;
            element?.Load();
        }
    }
}
