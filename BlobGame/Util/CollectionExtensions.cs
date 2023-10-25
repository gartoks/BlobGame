using System.Collections.Concurrent;

namespace BlobGame.Util;
public static class CollectionExtensions {

    public static void AddRange<T>(this ConcurrentBag<T> collection, IEnumerable<T> items) {
        foreach (T? item in items)
            collection.Add(item);
    }

}
