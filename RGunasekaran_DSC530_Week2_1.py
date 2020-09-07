# File :    RGunasekaran_DSC530_Week2_1.ipynb
# Name :    Ragunath Gunasekaran
# Date :    09/07/2020
# Course :  DSC-530 - Data Exploration and Analysis
# Assignment :
#           Display the text “Hello World! I wonder why that is always the default coding text to start with”
#           Add two numbers together
#           Subtract a number from another number
#           Multiply two numbers
#           Divide between two numbers
#           Concatenate two strings together (any words)
#           Create a list of 4 items (can be strings, numbers, both)
#           Append an item to your list (again, can be a string, number)
#           Create a tuple with 4 items (can be strings, numbers, both)
print("************************ First Print ************************************************")
# Display the text “Hello World! I wonder why that is always the default coding text to start with”
print("Hello World! I wonder why that is always the default coding text to start with")
print("************************ Arithmetic Operations ************************************************")
# Add two numbers together
# Subtract a number from another number
# Multiply two numbers
# Divide between two numbers

# Get the input values
input1 = int(input("Enter the first Number : "))
input2 = int(input("Enter the second Number : "))

# Select the Arthimetic Operation to be Performed
operationentry = int(
    input("Enter the Arithmetic Operation to be performed - 1.Add 2.Subtract 3.Multiply 4.Divide 5.modulo \n"))

if (operationentry == 1):  # add
    total = input1 + input2
elif (operationentry == 2):  # Subtract
    total = input1 - input2
elif (operationentry == 3):  # Multiply
    total = input1 * input2
elif (operationentry == 4):  # Divide
    total = input1 / input2
elif (operationentry == 5):  # modulo
    total = input1 % input2

print("The Value is: ", total)
print("************************ String Concatenation ************************************************")
# Concatenate two strings together (any words)
# Get the input strings
inputstr1 = str(input("Enter the First String : "))
inputstr2 = str(input("Enter the Second String : "))
# combine the strings by using join
concatenagestring = ("".join([inputstr1, inputstr2]))
print("Concatenation of the string is " + concatenagestring)
# combine the strings by using join with a separator Space(" ")
concatenagestring = " ".join([inputstr1, inputstr2])
print("Concatenation of the string with space is " + concatenagestring)
print("************************ list append ************************************************")
country_list = ['india', 'usa', 'uk', 'uae', 5, 6]
country_list.append(1)
country_list.append('dubai')
print(country_list)
print("************************ tuple Concatenation ************************************************")
# Create a tuple with 4 items (can be strings, numbers, both)
tuple1 = (0, 1, 2, 3)
tuple2 = ('python', 'r')
# Concatenating two tuples
print(tuple1 + tuple2)
# nesting of two tuples
print(tuple1, tuple2)
