module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    // No plugins needed for expo-router in SDK 50+
  };
};
