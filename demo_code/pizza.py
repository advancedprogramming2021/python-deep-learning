def make_pizza(size, *toppings):
    print('Making a " + str(size) + "-inch pizza with the following toppings:')
    for topping in toppings:
        print("- " + topping)
