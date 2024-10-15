// src/Main.tsx
import * as React from 'react';
import { View, StyleSheet } from 'react-native';
import { Button, Text } from 'react-native-paper';

const Main = () => {
  return (
    <View style={styles.container}>
      <Text variant="headlineMedium">Welcome to MyApp!</Text>
      <Button mode="contained" onPress={() => console.log('Button pressed')}>
        Press me
      </Button>
    </View>
  );
};

export default Main;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: 16,
  },
});