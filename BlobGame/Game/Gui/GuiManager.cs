﻿using BlobGame.App;
using BlobGame.Audio;
using OpenTK.Windowing.GraphicsLibraryFramework;

namespace BlobGame.Game.Gui;
/// <summary>
/// Class to keep track of gui elements and focus.
/// </summary>
internal static class GuiManager {
    private static List<GuiElement> Elements { get; }
    private static List<InteractiveGuiElement> InteractiveElements { get; }
    private static int FocusedElementIndex { get; set; }

    static GuiManager() {
        Elements = new();
        InteractiveElements = new();
        FocusedElementIndex = -1;
    }

    internal static void Initialize() {
    }

    internal static void Load() {
        Input.RegisterHotkey("confirm", Keys.Enter);
        Input.RegisterHotkey("next", Keys.Down);
        Input.RegisterHotkey("previous", Keys.Up);
        Input.RegisterHotkey("next_subItem", Keys.Right);
        Input.RegisterHotkey("previous_subItem", Keys.Left);
    }

    internal static void Unload() {
        ResetElements();
        Input.UnregisterHotkey("confirm");
        Input.UnregisterHotkey("next");
        Input.UnregisterHotkey("previous");
    }

    internal static void Update(float dT) {
        if (Input.IsHotkeyActive("next")) {
            do {
                FocusedElementIndex = (FocusedElementIndex + 1) % InteractiveElements.Count;
            } while (!InteractiveElements[FocusedElementIndex].Enabled);
            AudioManager.PlaySound("ui_interaction");
        } else if (Input.IsHotkeyActive("previous")) {
            do {
                FocusedElementIndex = (FocusedElementIndex - 1 + InteractiveElements.Count) % InteractiveElements.Count;
            } while (!InteractiveElements[FocusedElementIndex].Enabled);
            AudioManager.PlaySound("ui_interaction");
        }
    }

    internal static void ResetElements() {
        Elements.Clear();
        InteractiveElements.Clear();
        FocusedElementIndex = -1;
    }

    internal static void AddElement(GuiElement element) {
        if (!Elements.Contains(element))
            Elements.Add(element);
        if (element is InteractiveGuiElement interactiveGuiElement && !InteractiveElements.Contains(interactiveGuiElement))
            InteractiveElements.Add(interactiveGuiElement);
    }

    internal static void Focus(InteractiveGuiElement? element) {
        if (element is null) {
            FocusedElementIndex = -1;
            return;
        }

        if (!element.Enabled)
            return;

        if (HasFocus(element))
            return;

        int idx = InteractiveElements.IndexOf(element);
        if (idx == -1)
            return;

        AudioManager.PlaySound("ui_interaction");
        FocusedElementIndex = idx;
    }

    internal static bool HasFocus(InteractiveGuiElement element) {
        return FocusedElementIndex != -1 && InteractiveElements[FocusedElementIndex] == element;
    }
}
