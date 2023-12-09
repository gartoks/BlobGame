using System.Collections;

namespace BlobGame.Game.Util;

/// <summary>
/// Interface used as a base for the GameObjectCollection class. Only allows read-only operations.
/// All operations are thread-safe.
/// </summary>
public interface IReadOnlyGameObjectsCollection : IEnumerable<GameObject> {
    /// <summary>
    /// The number of game objects in this collection.
    /// </summary>
    int Count { get; }

    /// <summary>
    /// Finds a game object by name. Returns null if no object with the given name exists.
    /// </summary>
    /// <param name="name">The name of the game object to find.</param>
    /// <returns>Returns the game object if found; null otherwise.</returns>
    GameObject? FindByName(string name);
    /// <summary>
    /// Checks if the collection contains the given game object.
    /// </summary>
    /// <param name="item">The game object to check</param>
    /// <returns>Returns true of the game object exists in this collection; false otherwise.</returns>
    bool Contains(GameObject item);
    /// <summary>
    /// Enumerates over all game objects in this collection and executes the given action on each.
    /// This locks the collection for the duration of the enumeration.
    /// </summary>
    /// <param name="iterationAction">The action to be called for each game object.</param>
    void Enumerate(Action<GameObject> iterationAction);
}

/// <summary>
/// Class used to store and manage game objects. All operations are thread-safe.
/// </summary>
internal class GameObjectCollection : IReadOnlyGameObjectsCollection {
    /// <summary>
    /// The number of game objects in this collection.
    /// </summary>
    public int Count => List.Count;

    /// <summary>
    /// Internal list of game objects.
    /// </summary>
    private List<GameObject> List { get; }
    /// <summary>
    /// Lock object used to synchronize access to the list.
    /// </summary>
    private object Lock { get; }
    /// <summary>
    /// Flag used to determine if the list needs to be sorted before enumeration.
    /// </summary>
    private bool IsDirty { get; set; }

    /// <summary>
    /// Creates a new game object collection.
    /// </summary>
    public GameObjectCollection() {
        List = new List<GameObject>();
        Lock = new object();
        IsDirty = false;
    }

    /// <summary>
    /// Finds a game object by name. Returns null if no object with the given name exists.
    /// </summary>
    /// <param name="name">The name of the game object to find.</param>
    /// <returns>Returns the game object if found; null otherwise.</returns>
    public GameObject? FindByName(string name) {
        lock (Lock) {
            return List.SingleOrDefault(item => item.Name == name);
        }
    }

    /// <summary>
    /// Enumerates over all game objects in this collection and executes the given action on each.
    /// This locks the collection for the duration of the enumeration.
    /// </summary>
    /// <param name="iterationAction">The action to be called for each game object.</param>
    public void Enumerate(Action<GameObject> iterationAction) {
        lock (Lock) {
            if (IsDirty) {
                List.Sort((a, b) => a.ZIndex.CompareTo(b.ZIndex));
                IsDirty = false;
            }

            foreach (GameObject item in List) {
                iterationAction(item);
            }
        }
    }

    /// <summary>
    /// Adds a game object to the collection. This operation requires the collection to be re-sorted after adding.
    /// </summary>
    /// <param name="item"></param>
    public void Add(GameObject item) {
        lock (Lock) {
            List.Add(item);
            IsDirty = true;
        }
    }

    /// <summary>
    /// Adds a range of game objects to the collection. This operation requires the collection to be re-sorted after adding.
    /// </summary>
    /// <param name="en"></param>
    public void AddRange(IEnumerable<GameObject> en) {
        lock (Lock) {
            List.AddRange(en);
            IsDirty = true;
        }
    }

    /// <summary>
    /// Removes a game object from the collection. This operation keeps the collection sorted.
    /// </summary>
    /// <param name="item"></param>
    /// <returns></returns>
    public bool Remove(GameObject item) {
        lock (Lock) {
            return List.Remove(item);
        }
    }

    /// <summary>
    /// Removes all game objects from the collection.
    /// </summary>
    public void Clear() {
        lock (Lock) {
            List.Clear();
        }
    }

    /// <summary>
    /// Checks if the collection contains the given game object.
    /// </summary>
    /// <param name="item">The game object to check</param>
    /// <returns>Returns true of the game object exists in this collection; false otherwise.</returns>
    public bool Contains(GameObject item) {
        lock (Lock) {
            return List.Contains(item);
        }
    }

    public IEnumerator<GameObject> GetEnumerator() => List.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
