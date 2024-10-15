// HomeScreen.tsx
import React from 'react';
import { StyleSheet } from 'react-native';
import { Card, Button, Title, Paragraph, Text } from 'react-native-paper';
import { ScrollView, View } from 'react-native';
import { useTheme } from 'react-native-paper';

export default function HomeScreen() {
  const theme = useTheme();

  const coffeeBeans = [
    {
      name: 'Arabica',
      // image: require('@/assets/images/arabica.jpg'),
      description: 'Smooth and sweet with hints of chocolate and sugar.',
    },
    {
      name: 'Robusta',
      // image: require('@/assets/images/robusta.jpg'),
      description: 'Strong and bold with high caffeine content.',
    },
    {
      name: 'Liberica',
      // image: require('@/assets/images/liberica.jpg'),
      description: 'Unique flavor with floral and fruity notes.',
    },
    {
      name: 'Excelsa',
      // image: require('@/assets/images/excelsa.jpg'),
      description: 'Tart and fruity flavor profile.',
    },
  ];

  const handleSelectBean = (beanName: string) => {
    // Handle bean selection (e.g., navigate to another screen or perform an action)
    console.log(`Selected bean: ${beanName}`);
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        {/* Remove the logo image */}
        {/* <Image
          source={require('@/assets/images/opticroplogo.png')}
          style={styles.logo}
          resizeMode="contain"
        /> */}
        <Title style={[styles.title, { color: theme.colors.primary }]}>OptiCrop</Title>
      </View>

      <Text style={styles.subtitle}>Select a Coffee Bean Variety</Text>

      <View style={styles.cardContainer}>
        {coffeeBeans.map((bean, index) => (
          <Card style={styles.card} key={index}>
            {/* Remove the card cover image */}
            {/* <Card.Cover source={bean.image} /> */}
            <Card.Content>
              <Title>{bean.name}</Title>
              <Paragraph>{bean.description}</Paragraph>
            </Card.Content>
            <Card.Actions>
              <Button onPress={() => handleSelectBean(bean.name)}>Select</Button>
            </Card.Actions>
          </Card>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F1F8E9', // Light greenish background
  },
  header: {
    alignItems: 'center',
    marginTop: 24,
    marginBottom: 24,
  },
  // Remove the logo style
  // logo: {
  //   width: 100,
  //   height: 100,
  // },
  title: {
    fontSize: 32,
    marginTop: 8,
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: 20,
    marginBottom: 16,
    textAlign: 'center',
    color: '#3E2723',
  },
  cardContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
    paddingHorizontal: 8,
  },
  card: {
    width: '45%',
    marginBottom: 16,
    elevation: 4, // Adds shadow effect
  },
});
